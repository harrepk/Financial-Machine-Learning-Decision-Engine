"""Streamlit frontend for the Financial Decision Engine.

Run from the project root, with the API already running on port 8000:
    streamlit run frontend/app.py

The UI never touches the database or the ML models directly. It only packages the
user's inputs, sends them to the FastAPI backend over HTTP, and displays the JSON
that comes back. That separation is what "decoupled client-server" means.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

# 127.0.0.1 rather than "localhost": on Windows, "localhost" tries IPv6 first, which
# uvicorn doesn't listen on by default, adding ~0.25 s to every request.
API_URL = os.getenv("FDE_API_URL", "http://127.0.0.1:8000")
START_BACKEND_HINT = "uvicorn backend.main:app --port 8000"

TIER_NAMES = {0: "Starter", 1: "Standard", 2: "Premium", 3: "Elite"}
BAND_COLORS = {"High": "green", "Moderate": "orange", "Low": "red"}
EXAMPLE_SEARCHES = {  # button label -> search text
    "Travel, no FX fees": "no foreign transaction fee travel card",
    "Grocery cash back": "cash back on groceries with no annual fee",
    "Student starter": "student card to build credit",
    "Low interest": "low interest card to pay off debt",
}

st.set_page_config(page_title="Financial Decision Engine", page_icon="💳", layout="wide")

# Streamlit re-runs this whole script from top to bottom on every click. Anything that
# must survive a rerun (like the API's answer) is kept in st.session_state.
st.session_state.setdefault("query", EXAMPLE_SEARCHES["Travel, no FX fees"])
st.session_state.setdefault("result", None)
st.session_state.setdefault("error", None)


def money(amount: float) -> str:
    """Format dollars for Markdown. The $ is escaped because Streamlit treats $...$ as LaTeX math."""
    return f"\\${amount:,.0f}"


def use_example(search: str) -> None:
    st.session_state.query = search


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def fetch_recommendations(payload: dict) -> tuple[dict | None, str | None]:
    """POST to /recommend. Returns (result, None) on success or (None, friendly error message)."""
    try:
        response = requests.post(f"{API_URL}/recommend", json=payload, timeout=15)
    except requests.ConnectionError:
        return None, f"Can't reach the backend at {API_URL}. Start it in another terminal with:  `{START_BACKEND_HINT}`"
    except requests.Timeout:
        return None, "The backend took too long to answer. Please try again."

    if response.status_code == 422:  # FastAPI/Pydantic rejected an input
        problems = [f"{err['loc'][-1].replace('_', ' ')}: {err['msg']}" for err in response.json()["detail"]]
        return None, "Please check your inputs. " + "; ".join(problems)
    if not response.ok:
        detail = response.json().get("detail", response.text) if response.content else response.reason
        return None, f"Backend error {response.status_code}: {detail}"
    return response.json(), None


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("System status")
    health = fetch_health()
    if health is None:
        st.error(f"Backend offline at {API_URL}")
        st.code(START_BACKEND_HINT, language="bash")
    elif not health["predictor_loaded"]:
        st.warning("Backend online, but the model isn't trained yet.")
        st.code("python -m ml.train", language="bash")
    else:
        st.success(f"Backend online · {health['cards_loaded']} cards · model loaded")
        metrics = health.get("predictor_metrics") or {}
        if metrics:
            st.caption(
                f"Model validation accuracy {metrics['mlp_val_accuracy']:.1%}, "
                f"ROC-AUC {metrics['mlp_val_auc']:.3f}"
            )

    st.header("How it works")
    st.markdown(
        "1. **Matchmaker** reads your search and finds the 3 closest cards "
        "(TF-IDF + cosine similarity).\n"
        "2. **Predictor** feeds your numbers into a neural network that estimates your "
        "approval odds for each card (PyTorch).\n"
        "3. The **API** returns everything as JSON and this page displays it."
    )
    st.caption("All cards and banks are fictional. The model is trained on synthetic data.")


# ---------------- Main form ----------------
st.title("💳 Financial Decision Engine")
st.write(
    "Describe the credit card you want in plain English, add your numbers, and see the best "
    "matches together with your estimated chance of approval."
)

st.caption("Try an example:")
example_columns = st.columns(len(EXAMPLE_SEARCHES))
for column, (label, search) in zip(example_columns, EXAMPLE_SEARCHES.items(), strict=True):
    column.button(label, on_click=use_example, args=(search,), help=search)

with st.form("applicant_form"):
    st.text_input("What are you looking for in a card?", key="query", max_chars=200)
    col_score, col_income, col_debt = st.columns(3)
    credit_score = col_score.slider(
        "Credit score", min_value=300, max_value=900, value=700, step=5,
        help="Canadian credit scores range from 300 to 900.",
    )
    annual_income = col_income.number_input(
        "Annual income ($)", min_value=0, max_value=5_000_000, value=60_000, step=1_000,
        help="Your yearly income before tax.",
    )
    existing_debt = col_debt.number_input(
        "Existing debt ($)", min_value=0, max_value=5_000_000, value=10_000, step=500,
        help="Total of student loans, car loans, credit card balances, etc.",
    )
    submitted = st.form_submit_button("Find my cards", type="primary")

if submitted:
    payload = {
        "query": st.session_state.query,
        "credit_score": credit_score,
        "annual_income": annual_income,
        "existing_debt": existing_debt,
    }
    with st.spinner("Matching cards and scoring your application..."):
        st.session_state.result, st.session_state.error = fetch_recommendations(payload)


# ---------------- Results (read from session_state so they survive reruns) ----------------
if st.session_state.error:
    st.error(st.session_state.error)

result = st.session_state.result
if result:
    st.divider()
    st.subheader(f"Results for “{result['query']}”")

    if not result["recommendations"]:
        st.info(result["message"])

    for rank, rec in enumerate(result["recommendations"], start=1):
        card = rec["card"]
        band = rec["approval_band"]
        with st.container(border=True):
            details, odds = st.columns([3, 2], gap="large")
            with details:
                st.markdown(f"#### {rank}. {card['name']}")
                st.caption(f"{card['issuer']} · {card['category'].title()} · {TIER_NAMES[card['tier']]} tier")
                st.write(card["description"])
                st.markdown(f"**Rewards:** {card['rewards']}")
                annual_fee = "None" if card["annual_fee"] == 0 else money(card["annual_fee"])
                fx_fee = "None" if card["foreign_transaction_fee"] == 0 else f"{card['foreign_transaction_fee']:.1f}%"
                st.markdown(
                    f"**Annual fee:** {annual_fee} &nbsp;·&nbsp; **Purchase APR:** {card['purchase_apr']:.2f}% "
                    f"&nbsp;·&nbsp; **Foreign transaction fee:** {fx_fee}"
                )
            with odds:
                st.metric("Estimated approval odds", f"{rec['approval_probability']:.0%}")
                st.markdown(f":{BAND_COLORS[band]}[**{band} likelihood**]")
                st.progress(rec["approval_probability"])
                st.caption(f"Match with your search: {rec['match_score']:.0%}")
                if not rec["meets_income_guideline"]:
                    st.warning(f"Issuer suggests an income of at least {money(card['min_income'])}.")

    if result["insights"]:
        st.subheader("What your numbers say")
        for insight in result["insights"]:
            st.markdown(f"- {insight}")

    # Toggling this reruns the script. Because the result lives in session_state,
    # the cards above stay on screen instead of disappearing.
    if st.toggle("Show the raw JSON returned by the API"):
        st.json(result)

    st.caption(result["disclaimer"])
