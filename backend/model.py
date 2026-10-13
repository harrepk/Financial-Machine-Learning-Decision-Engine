"""The PyTorch approval model. Shared by training (ml/train.py) and serving (backend/predictor.py).

Keeping the feature recipe and the network in ONE file guarantees the model sees
exactly the same kind of numbers in production as it did during training. A mismatch
there ("training/serving skew") is one of the most common real-world ML bugs.
"""
from __future__ import annotations

import math

import torch
from torch import nn

FEATURE_NAMES = ("credit_score", "log_annual_income", "log_existing_debt", "card_tier")
N_FEATURES = len(FEATURE_NAMES)


def build_features(credit_score: float, annual_income: float, existing_debt: float, card_tier: int) -> list[float]:
    """Turn one applicant + one card into the model's 4-number feature vector.

    Income and debt are log-scaled because money is skewed: the jump from $20k to $40k
    changes someone's situation far more than the jump from $220k to $240k.
    """
    return [float(credit_score), math.log1p(annual_income), math.log1p(existing_debt), float(card_tier)]


class ApprovalMLP(nn.Module):
    """A 3-layer Multi-Layer Perceptron: 4 inputs -> 32 -> 16 -> 1 output.

    The output is a raw score (a "logit"). Passing it through a sigmoid squashes it
    into a probability between 0 and 1, e.g. 0.88 = 88% chance of approval.
    """

    def __init__(self, n_features: int = N_FEATURES, hidden: tuple[int, int] = (32, 16)):
        super().__init__()
        # Normalisation statistics are stored inside the model (as buffers, not trainable
        # weights) so the saved .pt file is fully self-contained.
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_std", torch.ones(n_features))
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1e-6))  # avoid dividing by zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x has shape [batch, n_features]. Returns logits of shape [batch, 1]."""
        return self.net((x - self.feature_mean) / self.feature_std)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid turns logits into approval probabilities in [0, 1]. Shape [batch, 1]."""
        return torch.sigmoid(self.forward(x))
