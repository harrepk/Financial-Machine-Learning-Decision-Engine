"""Shared pytest fixtures. Every test uses temporary files, so the real data/ and models/ are never touched."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import config
from backend.database import fetch_all_cards, init_db
from backend.main import create_app
from backend.matcher import CardMatcher
from backend.predictor import ApprovalPredictor
from ml.train import train_model


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    """Train a small model once per test run (takes a few seconds)."""
    path = tmp_path_factory.mktemp("models") / "approval_mlp.pt"
    train_model(rows=3_000, epochs=30, output_path=path, verbose=False)
    return path


@pytest.fixture(scope="session")
def predictor(model_path):
    return ApprovalPredictor(model_path)


@pytest.fixture(scope="session")
def cards(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "cards.db"
    init_db(db_path, config.CARDS_SEED_PATH)
    return fetch_all_cards(db_path)


@pytest.fixture(scope="session")
def matcher(cards):
    return CardMatcher(cards)


@pytest.fixture()
def client(tmp_path, model_path):
    app = create_app(db_path=tmp_path / "cards.db", seed_path=config.CARDS_SEED_PATH, model_path=model_path)
    with TestClient(app) as test_client:  # `with` runs the startup (lifespan) code
        yield test_client
