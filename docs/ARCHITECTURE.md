# Architecture

This document shows the Financial Decision Engine from several angles: the big
picture first, then each stage in depth. Every diagram is written in
[Mermaid](https://mermaid.js.org/), which GitHub renders automatically.

- [1. System overview](#1-system-overview)
- [2. Component view](#2-component-view)
- [3. Request lifecycle: one search, end to end](#3-request-lifecycle-one-search-end-to-end)
- [4. Stage 1: The Matchmaker](#4-stage-1-the-matchmaker)
- [5. Stage 2: The Predictor](#5-stage-2-the-predictor)
- [6. Offline training pipeline](#6-offline-training-pipeline)
- [7. Data model](#7-data-model)
- [8. Runtime view: processes and ports](#8-runtime-view-processes-and-ports)
- [9. Startup lifecycle](#9-startup-lifecycle)
- [10. Design decisions](#10-design-decisions)

---

## 1. System overview

The user talks to a Streamlit web page. The page talks to a FastAPI server. The
server owns the database and both ML models. Training happens offline, ahead of time.

```mermaid
flowchart LR
    user(["👤 User"])
    subgraph client["Frontend · Streamlit · port 8501"]
        ui["Search box + financial inputs<br/>Results cards + approval odds"]
    end
    subgraph server["Backend · FastAPI · port 8000"]
        api["REST API<br/>/match · /predict · /recommend"]
        s1["Stage 1: Matchmaker<br/>TF-IDF + cosine similarity"]
        s2["Stage 2: Predictor<br/>PyTorch MLP + sigmoid"]
    end
    db[("SQLite<br/>data/cards.db")]
    model[/"models/approval_mlp.pt"/]
    subgraph offline["Offline · run once"]
        train["ml/train.py<br/>synthetic data → trained model"]
    end

    user --> ui
    ui -- "HTTP POST JSON" --> api
    api --> s1 --> db
    api --> s2 --> model
    api -- "JSON response" --> ui
    train -- "saves" --> model
```

## 2. Component view

Each box is a Python module. Arrows mean "imports / calls".

```mermaid
flowchart TB
    subgraph frontend["frontend/"]
        app["app.py<br/>Streamlit UI, session_state"]
    end
    subgraph backend["backend/"]
        main["main.py<br/>FastAPI app factory, routes, lifespan"]
        schemas["schemas.py<br/>Pydantic request/response models"]
        database["database.py<br/>SQLite schema, seed, queries"]
        matcher["matcher.py<br/>CardMatcher (TF-IDF)"]
        predictor["predictor.py<br/>ApprovalPredictor (loads .pt)"]
        model["model.py<br/>ApprovalMLP + build_features()"]
        config["config.py<br/>paths, thresholds"]
    end
    subgraph ml["ml/"]
        gen["generate_data.py<br/>synthetic applications"]
        train["train.py<br/>training loop + baseline"]
    end

    app -- "HTTP only" --> main
    main --> schemas
    main --> database
    main --> matcher
    main --> predictor
    predictor --> model
    train --> model
    train --> gen
    main --> config
    database --> schemas
    matcher --> schemas
```

> `model.py` is shared by training **and** serving. The exact same `build_features()`
> function prepares numbers in both places, which prevents training/serving skew.

## 3. Request lifecycle: one search, end to end

What happens when the user clicks **Find my cards**:

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant S as Streamlit (8501)
    participant A as FastAPI (8000)
    participant M as CardMatcher (TF-IDF)
    participant P as ApprovalPredictor (PyTorch)

    U->>S: types "no foreign transaction fee travel card"<br/>score 720, income $65k, debt $8k
    S->>A: POST /recommend {query, credit_score, annual_income, existing_debt}
    A->>A: Pydantic validates inputs (422 if invalid)
    A->>M: search(query, top_k=3)
    M->>M: query → TF-IDF vector → cosine similarity vs 20 cards
    M-->>A: top 3 cards + match scores
    A->>P: predict(720, 65000, 8000, card_tiers=[1, 2, 3])
    P->>P: build 3×4 feature tensor → MLP → sigmoid
    P-->>A: [0.95, 0.83, 0.60]
    A->>A: add bands (High/Moderate/Low), income check, insights
    A-->>S: 200 OK · RecommendResponse JSON
    S->>S: store JSON in st.session_state
    S-->>U: 3 result cards with approval odds
```

## 4. Stage 1: The Matchmaker

**Goal:** turn a plain-English wish into the three most relevant cards.

```mermaid
flowchart TD
    start(["User submits a search"]) --> validate{"Query is 2–200<br/>characters?"}
    validate -- No --> err422["Return HTTP 422<br/>UI shows 'check your inputs'"]
    validate -- Yes --> analyze["Clean the text<br/>lowercase · split words · drop stop words<br/>(but keep 'no') · fees→fee · add bigrams"]
    analyze --> vector["Turn it into a TF-IDF vector<br/>using the vectorizer fitted at startup"]
    vector --> known{"Any word known<br/>to the catalogue?"}
    known -- No --> nomatch["Return empty list +<br/>'try words like travel, cash back…'"]
    known -- Yes --> cosine["Cosine similarity vs.<br/>every card vector"]
    cosine --> rank["Sort scores high → low<br/>keep top 3"]
    rank --> threshold{"Score ≥ 0.05?"}
    threshold -- "No (weak match)" --> drop["Drop that card"]
    threshold -- Yes --> keep["Keep card + match score"]
    keep --> out(["Matches handed to Stage 2"])
    drop --> out
```

How a card becomes searchable text (`card_document()` in `matcher.py`):

| Source field | Example | Why |
|---|---|---|
| name | "Lakeshore Passport Card" | users sometimes search by name |
| category tag | "travel" | strong signal of purpose |
| derived tags | "no foreign transaction fee", "no annual fee", "low interest rate" | turns numeric columns into searchable words |
| rewards | "2x points on travel, gas and transit" | what the user earns |
| description | full marketing text | everything else |

## 5. Stage 2: The Predictor

**Goal:** estimate the probability that this applicant is approved for each matched card.

```mermaid
flowchart TD
    start(["Applicant numbers + 3 matched cards"]) --> valid{"Valid numbers?<br/>score 300–900 · income > 0 · debt ≥ 0"}
    valid -- No --> err["HTTP 422 → friendly error in UI"]
    valid -- Yes --> loaded{"Model file loaded<br/>at startup?"}
    loaded -- No --> e503["HTTP 503<br/>'run python -m ml.train'"]
    loaded -- Yes --> feats["build_features() for each card<br/>[score, log(income), log(debt), card tier]"]
    feats --> tensor["Stack into a 2D tensor<br/>shape [3, 4] = 3 cards × 4 features"]
    tensor --> norm["Normalise with the training<br/>mean and std (stored in the model)"]
    norm --> mlp["MLP forward pass<br/>4 → 32 → 16 → 1 with ReLU"]
    mlp --> sigmoid["Sigmoid squashes each logit<br/>into a probability 0–1"]
    sigmoid --> band["Label: High ≥ 70% · Moderate ≥ 40% · Low"]
    band --> out(["JSON: probability + band per card"])
```

The network in detail:

```mermaid
flowchart LR
    subgraph input["Input layer (4)"]
        i1["credit score"]
        i2["log income"]
        i3["log debt"]
        i4["card tier"]
    end
    subgraph h1["Hidden layer 1 (32, ReLU)"]
        a["32 neurons"]
    end
    subgraph h2["Hidden layer 2 (16, ReLU)"]
        b["16 neurons"]
    end
    subgraph out["Output (1)"]
        o["logit → sigmoid → P(approved)"]
    end
    i1 & i2 & i3 & i4 --> a --> b --> o
```

Why `card tier` is an input: the original plan fed only the 3 applicant numbers
(shape `[1, 3]`), which would give the **same** probability for every matched card.
Adding the card's tier (Starter → Elite) lets one applicant get different odds for
different cards, and all three cards are scored in a single batched forward pass
(shape `[3, 4]`).

## 6. Offline training pipeline

Run once with `python -m ml.train`. Takes a few seconds on a laptop CPU.

```mermaid
flowchart TD
    gen["generate_data.py<br/>5,000 synthetic applications"] --> policy["Hidden 'bank policy'<br/>score, income, debt-to-income, tier<br/>+ coin flip for realism"]
    policy --> feats["build_features()<br/>(same function the API uses)"]
    feats --> split["Stratified split<br/>80% train · 20% validation"]
    split --> normstats["Compute mean/std on TRAIN only<br/>store inside the model"]
    normstats --> loop["Training loop · Adam · BCEWithLogitsLoss<br/>batches of 64 · 60 epochs"]
    loop --> best["Keep the epoch with the<br/>lowest validation loss"]
    best --> eval["Evaluate: accuracy + ROC-AUC<br/>vs. logistic regression baseline"]
    eval --> save[/"models/approval_mlp.pt<br/>models/approval_mlp.metrics.json"/]
```

## 7. Data model

```mermaid
erDiagram
    CREDIT_CARDS {
        INTEGER id PK
        TEXT name "unique"
        TEXT issuer
        TEXT category "travel, cashback, student, ..."
        INTEGER tier "0 Starter to 3 Elite"
        REAL annual_fee
        REAL purchase_apr "percent"
        REAL foreign_transaction_fee "percent, 0 = none"
        INTEGER min_income "suggested minimum"
        TEXT rewards
        TEXT description
    }
```

The table is created and seeded from `data/credit_cards.json` on first startup.
All 20 cards and their banks are **fictional**.

## 8. Runtime view: processes and ports

```mermaid
flowchart LR
    subgraph t1["Terminal 1"]
        uv["uvicorn backend.main:app<br/>localhost:8000"]
    end
    subgraph t2["Terminal 2"]
        st["streamlit run frontend/app.py<br/>localhost:8501"]
    end
    browser(["Browser"]) -- "opens" --> st
    st -- "requests.post(http://localhost:8000/recommend)" --> uv
    uv -- "reads" --> files[("cards.db · approval_mlp.pt")]
```

The frontend reads the backend address from the `FDE_API_URL` environment
variable (default `http://localhost:8000`), so the two can later be deployed on
different machines without code changes.

## 9. Startup lifecycle

FastAPI's `lifespan` hook loads everything **once**. Requests then reuse it from memory.

```mermaid
flowchart TD
    boot(["uvicorn starts"]) --> init["init_db(): create table,<br/>seed 20 cards if empty"]
    init --> fetch["Read all cards into memory"]
    fetch --> fit["Fit TF-IDF vectorizer on card text"]
    fit --> load{"approval_mlp.pt exists?"}
    load -- Yes --> ready["Load weights · model.eval()"]
    load -- No --> warn["Log warning · predictor = None<br/>(/predict and /recommend → 503)"]
    ready --> serve(["Serve requests"])
    warn --> serve
```

## 10. Design decisions

| Decision | Alternative | Why this choice |
|---|---|---|
| Streamlit frontend | React | Pure Python, interactive UI in ~200 lines; the focus of this project is the backend/ML. |
| FastAPI backend | Flask | Built-in request validation (Pydantic), async support and auto-generated docs at `/docs`. |
| SQLite | PostgreSQL | Zero setup, a real SQL database in one file; easy to swap later because all SQL lives in `database.py`. |
| TF-IDF search | Sentence embeddings | Fast, explainable, no GPU or model download; good enough for a 20-card catalogue. |
| PyTorch MLP | Logistic regression | Learns non-linear rules (e.g. premium cards vs. subprime scores). The baseline is still trained and reported so the comparison is honest. |
| Card tier as a model input | Only 3 applicant features | Gives each card its own probability; demonstrates batched inference. |
| Normalisation stored in the model | Separate scaler file | One self-contained `.pt` artifact; impossible to load the wrong scaler. |
| `torch.load(weights_only=True)` | Default pickle loading | Loading a model file cannot execute arbitrary code. |
| Load models at startup | Load per request | Requests stay fast (milliseconds instead of re-reading files each time). |
| App factory `create_app()` | Global app only | Tests spin up isolated apps with temporary databases and models. |
