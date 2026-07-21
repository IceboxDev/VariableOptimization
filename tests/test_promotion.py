"""Promotion gate: one test per decision-table row, plus apply_promotion."""

import json

from variableoptimization import constants
from variableoptimization.promotion import (
    apply_promotion,
    decide_promotion,
    load_status_quo,
    status_quo_path,
)
from variableoptimization.runs import create_run_dir

FINGERPRINT = "a" * 16
OTHER_FINGERPRINT = "b" * 16


def baseline(best_loss: float = 100.0, fingerprint: str = FINGERPRINT) -> dict:
    return {
        "run_id": "20260721_090000_aaaaaaa",
        "model_sha256": "0" * 64,
        "metrics": {"best_loss": best_loss},
        "dataset_fingerprint": fingerprint,
        "generated_at": "2026-07-21T09:00:00+00:00",
    }


def test_row1_no_status_quo_is_initial_baseline():
    decision = decide_promotion(None, FINGERPRINT, best_loss=100.0)
    assert decision.promote
    assert decision.reason == "initial baseline"
    assert not decision.comparable


def test_row2_same_dataset_improved_loss_promotes():
    decision = decide_promotion(baseline(100.0), FINGERPRINT, best_loss=90.0)
    assert decision.promote
    assert "improved" in decision.reason
    assert decision.comparable
    assert decision.baseline_loss == 100.0


def test_row3_same_dataset_worse_or_equal_loss_rejects():
    for loss in (110.0, 100.0):  # equal counts as no improvement (strict <)
        decision = decide_promotion(baseline(100.0), FINGERPRINT, best_loss=loss)
        assert not decision.promote
        assert "no improvement" in decision.reason
        assert decision.comparable


def test_row4_changed_dataset_resets_baseline():
    decision = decide_promotion(baseline(100.0), OTHER_FINGERPRINT, best_loss=999.0)
    assert decision.promote
    assert "dataset changed" in decision.reason
    assert not decision.comparable


def test_apply_promotion_deploys_and_rewrites_baseline(tmp_path):
    output_dir = tmp_path
    paths = create_run_dir(output_dir / constants.RUNS_DIRNAME, "20260721_120000_ab12cd3")
    paths.model_path.write_bytes(b"model-bytes")
    paths.roster_path.write_text(json.dumps(["Alice", "Bob"]))

    metrics = {"best_loss": 90.0, "mean_loss": 95.0, "std_loss": 2.0}
    apply_promotion(paths, output_dir, "20260721_120000_ab12cd3", FINGERPRINT, metrics)

    assert (output_dir / constants.DEPLOYED_MODEL_FILENAME).read_bytes() == b"model-bytes"
    assert json.loads(
        (output_dir / constants.DEPLOYED_ROSTER_FILENAME).read_text()
    ) == ["Alice", "Bob"]

    status_quo = load_status_quo(output_dir)
    assert status_quo["run_id"] == "20260721_120000_ab12cd3"
    assert status_quo["dataset_fingerprint"] == FINGERPRINT
    assert status_quo["metrics"] == metrics


def test_no_promotion_leaves_status_quo_untouched(tmp_path):
    status_quo_path(tmp_path).write_text(json.dumps(baseline(100.0)))
    before = status_quo_path(tmp_path).read_text()

    decision = decide_promotion(load_status_quo(tmp_path), FINGERPRINT, best_loss=110.0)
    assert not decision.promote
    # The caller simply skips apply_promotion on a negative decision.
    assert status_quo_path(tmp_path).read_text() == before


def test_corrupt_status_quo_treated_as_absent(tmp_path, caplog):
    status_quo_path(tmp_path).write_text("{ not json")
    with caplog.at_level("WARNING"):
        assert load_status_quo(tmp_path) is None
