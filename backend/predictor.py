"""Stage 2 - The Predictor: load the trained PyTorch model once and score applicants with it."""
from __future__ import annotations

from pathlib import Path

import torch

from backend.model import N_FEATURES, ApprovalMLP, build_features


class ModelNotTrainedError(RuntimeError):
    """Raised when models/approval_mlp.pt does not exist yet."""


class ApprovalPredictor:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise ModelNotTrainedError(
                f"No trained model at {model_path}. Train one first with:  python -m ml.train"
            )
        # weights_only=True refuses to unpickle arbitrary Python objects, so a tampered
        # .pt file cannot run code on the server.
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model = ApprovalMLP(n_features=checkpoint["n_features"], hidden=tuple(checkpoint["hidden"]))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()  # inference mode: no training behaviour
        self.metrics: dict[str, float] = checkpoint.get("metrics", {})

    @staticmethod
    def to_tensor(rows: list[list[float]]) -> torch.Tensor:
        """Convert feature rows into the 2D [batch, features] tensor the network expects.

        This guards the classic PyTorch gotcha: a single row of features is 1D (shape [4])
        but nn.Linear needs 2D (shape [1, 4]), so we add the batch dimension explicitly.
        """
        x = torch.tensor(rows, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [4] -> [1, 4]
        if x.ndim != 2 or x.shape[1] != N_FEATURES:
            raise ValueError(f"Expected a tensor of shape [batch, {N_FEATURES}], got {list(x.shape)}")
        return x

    def predict(
        self, credit_score: float, annual_income: float, existing_debt: float, card_tiers: list[int]
    ) -> list[float]:
        """Return one approval probability per card, all scored in a single batched forward pass."""
        if not card_tiers:
            return []
        rows = [build_features(credit_score, annual_income, existing_debt, tier) for tier in card_tiers]
        x = self.to_tensor(rows)  # shape [n_cards, 4]
        with torch.inference_mode():  # no gradient tracking: faster and uses less memory
            probabilities = self.model.predict_proba(x).squeeze(1)  # [n_cards, 1] -> [n_cards]
        return probabilities.tolist()
