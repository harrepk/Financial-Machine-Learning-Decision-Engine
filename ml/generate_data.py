"""Generate a synthetic dataset of credit card applications.

Real approval data is private, so we simulate it. Each row is one applicant applying
for one card. A hidden "bank policy" (approval_logit below) turns the applicant's
numbers into an approval chance, then a weighted coin flip decides the outcome.
That coin flip matters: like real life, two similar people can get different
answers, so the model has to learn probabilities rather than memorise rules.

Usage (from the project root):
    python -m ml.generate_data --rows 5000 --out data/synthetic_applications.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import PROJECT_ROOT


def approval_logit(
    credit_score: np.ndarray, annual_income: np.ndarray, existing_debt: np.ndarray, card_tier: np.ndarray
) -> np.ndarray:
    """The hidden bank policy. Higher = more likely approved. The model never sees this formula."""
    debt_to_income = np.minimum(existing_debt / annual_income, 2.0)
    z = (
        0.02 * (credit_score - 680)  # every 50 points of credit score matters a lot
        + 1.2 * np.log(annual_income / 60_000)  # income helps, with diminishing returns
        - 3.0 * (debt_to_income - 0.40)  # debt relative to income hurts
        - 1.0 * (card_tier - 1.5)  # premium cards are harder to get
        + 0.3
    )
    # A non-linear rule: premium and elite cards are almost never given to subprime applicants.
    return z - 2.5 * ((credit_score < 560) & (card_tier >= 2))


def generate_applications(rows: int = 5_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    credit_score = np.clip(rng.normal(680, 85, rows), 300, 900).round()
    annual_income = np.clip(rng.lognormal(np.log(60_000), 0.55, rows), 12_000, 500_000).round(-2)
    debt_ratio = rng.gamma(shape=1.5, scale=0.25, size=rows)  # most people owe 10-60% of income
    debt_ratio[rng.random(rows) < 0.10] = 0.0  # about 10% have no debt at all
    existing_debt = (annual_income * debt_ratio).round(-2)
    card_tier = rng.integers(0, 4, rows)

    probability = 1 / (1 + np.exp(-approval_logit(credit_score, annual_income, existing_debt, card_tier)))
    approved = (rng.random(rows) < probability).astype(int)

    return pd.DataFrame(
        {
            "credit_score": credit_score.astype(int),
            "annual_income": annual_income,
            "existing_debt": existing_debt,
            "card_tier": card_tier,
            "approved": approved,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "synthetic_applications.csv")
    args = parser.parse_args()

    df = generate_applications(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out} (approval rate {df['approved'].mean():.1%})")


if __name__ == "__main__":
    main()
