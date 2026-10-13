"""Train the approval neural network and save it as models/approval_mlp.pt.

Usage (from the project root):
    python -m ml.train                          # 5,000 rows, 60 epochs (a few seconds on a laptop CPU)
    python -m ml.train --rows 20000 --epochs 100

Steps:
1. Generate synthetic applications (ml/generate_data.py).
2. Convert each row into features with the SAME build_features() the API uses.
3. Split 80% train / 20% validation (validation data is never trained on).
4. Train the MLP with binary cross-entropy loss and the Adam optimiser.
5. Keep the epoch with the lowest validation loss (simple early stopping).
6. Compare against a logistic regression baseline so we know the network earns its keep.
7. Save the weights + metadata to a .pt file, and the metrics to a .json file.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backend.config import MODEL_PATH, PROJECT_ROOT
from backend.model import FEATURE_NAMES, N_FEATURES, ApprovalMLP, build_features
from ml.generate_data import generate_applications

HIDDEN = (32, 16)

# Three example applicants, all applying for a Standard (tier 1) card.
PERSONAS = {
    "Strong   (score 780, $95k income, $5k debt)": (780, 95_000, 5_000, 1),
    "Average  (score 680, $55k income, $15k debt)": (680, 55_000, 15_000, 1),
    "Stretched(score 590, $32k income, $28k debt)": (590, 32_000, 28_000, 1),
}


def to_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.array(
        [build_features(r.credit_score, r.annual_income, r.existing_debt, r.card_tier) for r in df.itertuples()],
        dtype=np.float32,
    )


def train_model(
    rows: int = 5_000,
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    output_path: Path = MODEL_PATH,
    verbose: bool = True,
) -> dict[str, float]:
    torch.manual_seed(seed)
    log = print if verbose else (lambda *args, **kwargs: None)

    # 1-3. Data -> features -> train/validation split
    df = generate_applications(rows, seed)
    X = to_feature_matrix(df)
    y = df["approved"].to_numpy(dtype=np.float32)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    log(f"Data: {len(X_train):,} train / {len(X_val):,} validation rows, approval rate {y.mean():.1%}")

    # 4. Model, loss, optimiser
    model = ApprovalMLP(N_FEATURES, HIDDEN)
    model.set_normalization(torch.from_numpy(X_train.mean(axis=0)), torch.from_numpy(X_train.std(axis=0)))
    # BCEWithLogitsLoss = sigmoid + binary cross-entropy in one numerically stable step.
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1)),  # labels: [N] -> [N, 1]
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).unsqueeze(1)

    # 5. Training loop with best-epoch checkpointing
    best_val_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()  # clear gradients from the previous batch
            loss = loss_fn(model(xb), yb)  # forward pass + how wrong we were
            loss.backward()  # backpropagation: compute gradients
            optimizer.step()  # nudge the weights downhill

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            log(f"  epoch {epoch:3d}/{epochs}  validation loss {val_loss:.4f}")
    model.load_state_dict(best_state)
    model.eval()

    # 6. Evaluate, and compare with a logistic regression baseline on the same features
    with torch.no_grad():
        val_prob = model.predict_proba(X_val_t).squeeze(1).numpy()
    baseline = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
    base_prob = baseline.predict_proba(X_val)[:, 1]
    metrics = {
        "mlp_val_accuracy": round(float(accuracy_score(y_val, val_prob >= 0.5)), 4),
        "mlp_val_auc": round(float(roc_auc_score(y_val, val_prob)), 4),
        "logreg_val_accuracy": round(float(accuracy_score(y_val, base_prob >= 0.5)), 4),
        "logreg_val_auc": round(float(roc_auc_score(y_val, base_prob)), 4),
        "best_val_loss": round(best_val_loss, 4),
        "best_epoch": best_epoch,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
    }

    # 7. Save a self-contained checkpoint (weights + normalisation buffers + metadata)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": N_FEATURES,
            "hidden": list(HIDDEN),
            "feature_names": list(FEATURE_NAMES),
            "metrics": metrics,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        output_path,
    )
    output_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    log(f"\nBest epoch: {best_epoch}  (validation loss {best_val_loss:.4f})")
    log(f"Neural network  -> accuracy {metrics['mlp_val_accuracy']:.1%}, ROC-AUC {metrics['mlp_val_auc']:.3f}")
    log(f"LogReg baseline -> accuracy {metrics['logreg_val_accuracy']:.1%}, ROC-AUC {metrics['logreg_val_auc']:.3f}")
    log("\nSample predictions for a Standard (tier 1) card:")
    with torch.no_grad():
        for label, persona in PERSONAS.items():
            p = model.predict_proba(torch.tensor([build_features(*persona)])).item()  # shape [1, 4]
            log(f"  {label}: {p:.0%}")
    log(f"\nSaved model to {display_path(output_path)}")
    return metrics


def display_path(path: Path) -> str:
    """Show paths relative to the project so logs don't expose local folder names."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    train_model(args.rows, args.epochs, args.batch_size, args.lr, args.seed, args.out)


if __name__ == "__main__":
    main()
