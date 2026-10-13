"""Pydantic models: the exact shape of every request and response the API accepts or returns.

FastAPI uses these to validate input automatically. If a user sends a credit score of
1200 or a negative income, the request is rejected with HTTP 422 before any of our
code runs, which is the "Valid numbers?" check in the Stage 2 flowchart.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.config import DEFAULT_TOP_K, DISCLAIMER

ApprovalBand = Literal["High", "Moderate", "Low"]


class Card(BaseModel):
    """One row of the credit_cards table."""

    id: int
    name: str
    issuer: str
    category: str
    tier: int = Field(ge=0, le=3, description="0 = Starter, 1 = Standard, 2 = Premium, 3 = Elite")
    annual_fee: float
    purchase_apr: float = Field(description="Annual interest rate on purchases, in percent")
    foreign_transaction_fee: float = Field(description="Fee on purchases in other currencies, in percent")
    min_income: int = Field(description="Suggested minimum annual income, in dollars")
    rewards: str
    description: str


class Applicant(BaseModel):
    """The user's financial numbers. These are the model's inputs."""

    credit_score: int = Field(..., ge=300, le=900, description="Canadian credit score (300-900)")
    annual_income: float = Field(..., gt=0, le=5_000_000, description="Yearly income before tax, in dollars")
    existing_debt: float = Field(..., ge=0, le=5_000_000, description="Total current debt, in dollars")


# ---------- Stage 1: /match ----------

class CardMatch(BaseModel):
    card: Card
    match_score: float = Field(description="Cosine similarity between the search and the card (0-1)")


class MatchResponse(BaseModel):
    query: str
    matches: list[CardMatch]
    message: str


# ---------- Stage 2: /predict ----------

class PredictRequest(Applicant):
    card_ids: list[int] = Field(..., min_length=1, max_length=10)


class CardPrediction(BaseModel):
    card_id: int
    approval_probability: float = Field(ge=0, le=1)
    approval_band: ApprovalBand


class PredictResponse(BaseModel):
    predictions: list[CardPrediction]


# ---------- Full pipeline: /recommend ----------

class RecommendRequest(Applicant):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=2, max_length=200, examples=["no foreign transaction fee travel card"])
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=5)


class Recommendation(BaseModel):
    card: Card
    match_score: float
    approval_probability: float = Field(ge=0, le=1)
    approval_band: ApprovalBand
    meets_income_guideline: bool


class RecommendResponse(BaseModel):
    query: str
    applicant: Applicant
    debt_to_income: float
    recommendations: list[Recommendation]
    insights: list[str]
    message: str
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: str
    cards_loaded: int
    predictor_loaded: bool
    predictor_metrics: dict[str, float] | None = None
