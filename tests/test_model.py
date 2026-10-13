import math

import pytest
import torch

from backend.model import N_FEATURES, ApprovalMLP, build_features
from backend.predictor import ApprovalPredictor, ModelNotTrainedError


def test_build_features_log_scales_money():
    features = build_features(700, 60_000, 0, 2)

    assert len(features) == N_FEATURES
    assert features == [700.0, pytest.approx(math.log1p(60_000)), 0.0, 2.0]


def test_mlp_output_shape():
    model = ApprovalMLP()
    x = torch.randn(5, N_FEATURES)  # 5 applicants, 4 features each

    assert model(x).shape == (5, 1)
    probabilities = model.predict_proba(x)
    assert torch.all((probabilities > 0) & (probabilities < 1))


def test_to_tensor_adds_missing_batch_dimension():
    single_row = build_features(700, 60_000, 10_000, 1)

    assert ApprovalPredictor.to_tensor(single_row).shape == (1, N_FEATURES)  # [4] -> [1, 4]


def test_to_tensor_rejects_wrong_number_of_features():
    with pytest.raises(ValueError, match="shape"):
        ApprovalPredictor.to_tensor([[700, 60_000, 10_000]])  # only 3 features: the classic shape bug


def test_missing_model_file_gives_helpful_error(tmp_path):
    with pytest.raises(ModelNotTrainedError, match="python -m ml.train"):
        ApprovalPredictor(tmp_path / "missing.pt")


def test_trained_model_beats_chance(predictor):
    assert predictor.metrics["mlp_val_auc"] > 0.80


def test_predictions_are_one_probability_per_card(predictor):
    probabilities = predictor.predict(700, 60_000, 10_000, card_tiers=[0, 1, 2])

    assert len(probabilities) == 3
    assert all(0 <= p <= 1 for p in probabilities)
    assert predictor.predict(700, 60_000, 10_000, card_tiers=[]) == []


def test_model_learned_sensible_directions(predictor):
    """Sanity checks a lender would expect: better credit helps, more debt hurts, premium cards are harder."""

    def odds(score=680, income=60_000, debt=15_000, tier=1):
        return predictor.predict(score, income, debt, [tier])[0]

    assert odds(score=800) > odds(score=560)
    assert odds(debt=2_000) > odds(debt=60_000)
    assert odds(income=120_000) > odds(income=25_000)
    assert odds(tier=0) > odds(tier=3)
