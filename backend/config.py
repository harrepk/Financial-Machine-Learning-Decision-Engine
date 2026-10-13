"""Central configuration: file paths and tunable constants.

Every path can be overridden with an environment variable, so tests and
deployments can point the app at different files without code changes.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = Path(os.getenv("FDE_DB_PATH", PROJECT_ROOT / "data" / "cards.db"))
CARDS_SEED_PATH = Path(os.getenv("FDE_CARDS_SEED_PATH", PROJECT_ROOT / "data" / "credit_cards.json"))
MODEL_PATH = Path(os.getenv("FDE_MODEL_PATH", PROJECT_ROOT / "models" / "approval_mlp.pt"))

# Stage 1 (Matchmaker): a card needs at least this cosine similarity to count as a match.
MIN_MATCH_SCORE = 0.05
DEFAULT_TOP_K = 3

# Stage 2 (Predictor): probability cut-offs for the plain-English labels.
HIGH_APPROVAL = 0.70
MODERATE_APPROVAL = 0.40

DISCLAIMER = (
    "Educational project. Cards and banks are fictional, and approval odds come from a model "
    "trained on synthetic data. This is not financial advice or a real lender decision."
)
