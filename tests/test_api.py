from fastapi.testclient import TestClient

from backend import config
from backend.main import create_app

APPLICANT = {"credit_score": 720, "annual_income": 65_000, "existing_debt": 8_000}


def test_health(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["cards_loaded"] == 20
    assert body["predictor_loaded"] is True
    assert "mlp_val_auc" in body["predictor_metrics"]


def test_list_cards(client):
    cards = client.get("/cards").json()

    assert len(cards) == 20
    assert {"name", "issuer", "tier", "annual_fee", "description"} <= cards[0].keys()


def test_match_endpoint(client):
    body = client.get("/match", params={"query": "student card to build credit", "top_k": 2}).json()

    assert len(body["matches"]) == 2
    assert body["matches"][0]["card"]["category"] == "student"


def test_predict_endpoint(client):
    response = client.post("/predict", json={**APPLICANT, "card_ids": [1, 8]})

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [p["card_id"] for p in predictions] == [1, 8]
    assert all(0 <= p["approval_probability"] <= 1 for p in predictions)
    assert all(p["approval_band"] in {"High", "Moderate", "Low"} for p in predictions)


def test_predict_unknown_card_is_404(client):
    response = client.post("/predict", json={**APPLICANT, "card_ids": [999]})
    assert response.status_code == 404


def test_recommend_full_pipeline(client):
    response = client.post("/recommend", json={**APPLICANT, "query": "no foreign transaction fee travel card"})

    assert response.status_code == 200
    body = response.json()
    recs = body["recommendations"]
    assert len(recs) == 3
    assert [r["match_score"] for r in recs] == sorted((r["match_score"] for r in recs), reverse=True)
    assert all(0 <= r["approval_probability"] <= 1 for r in recs)
    assert body["debt_to_income"] == round(8_000 / 65_000, 4)
    assert body["insights"]
    assert body["disclaimer"]


def test_recommend_with_no_matching_cards(client):
    response = client.post("/recommend", json={**APPLICANT, "query": "qwzx plorf"})

    assert response.status_code == 200
    assert response.json()["recommendations"] == []
    assert "No cards matched" in response.json()["message"]


def test_recommend_rejects_invalid_numbers(client):
    bad_inputs = [
        {**APPLICANT, "credit_score": 950},  # above the 900 maximum
        {**APPLICANT, "annual_income": 0},  # must be greater than 0
        {**APPLICANT, "existing_debt": -5},  # cannot be negative
    ]
    for payload in bad_inputs:
        response = client.post("/recommend", json={**payload, "query": "travel"})
        assert response.status_code == 422, payload


def test_api_without_trained_model_returns_503(tmp_path):
    app = create_app(db_path=tmp_path / "cards.db", seed_path=config.CARDS_SEED_PATH, model_path=tmp_path / "none.pt")
    with TestClient(app) as client:
        assert client.get("/health").json()["predictor_loaded"] is False
        response = client.post("/recommend", json={**APPLICANT, "query": "travel"})
        assert response.status_code == 503
        assert "python -m ml.train" in response.json()["detail"]
