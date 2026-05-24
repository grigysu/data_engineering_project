"""Unit tests for the train resume / schema-drift guard."""

from __future__ import annotations

import pytest

from ml.train import validate_resume_checkpoint


FEATS = ("a", "b", "c")


def test_resume_validation_accepts_matching_checkpoint():
    prev = {
        "feature_columns": list(FEATS),
        "hyperparams": {"hidden_size": 64, "num_layers": 2},
        "epoch": 12,
        "best_val_mse": 0.5,
    }
    start_epoch, best_val = validate_resume_checkpoint(prev, FEATS, 64, 2)
    assert start_epoch == 13
    assert best_val == 0.5


def test_resume_validation_defaults_when_metadata_missing():
    prev = {
        "feature_columns": list(FEATS),
        "hyperparams": {"hidden_size": 64, "num_layers": 2},
    }
    start_epoch, best_val = validate_resume_checkpoint(prev, FEATS, 64, 2)
    assert start_epoch == 1  # epoch 0 + 1
    assert best_val == float("inf")


def test_resume_rejects_feature_column_drift():
    prev = {
        "feature_columns": ["a", "b"],  # missing 'c'
        "hyperparams": {"hidden_size": 64, "num_layers": 2},
    }
    with pytest.raises(SystemExit, match="feature_columns mismatch"):
        validate_resume_checkpoint(prev, FEATS, 64, 2)


def test_resume_rejects_hidden_size_drift():
    prev = {
        "feature_columns": list(FEATS),
        "hyperparams": {"hidden_size": 32, "num_layers": 2},
    }
    with pytest.raises(SystemExit, match="hyperparams mismatch"):
        validate_resume_checkpoint(prev, FEATS, 64, 2)


def test_resume_rejects_num_layers_drift():
    prev = {
        "feature_columns": list(FEATS),
        "hyperparams": {"hidden_size": 64, "num_layers": 1},
    }
    with pytest.raises(SystemExit, match="hyperparams mismatch"):
        validate_resume_checkpoint(prev, FEATS, 64, 2)
