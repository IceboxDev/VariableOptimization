"""Promotion gate: decides whether a training run becomes the deployed model.

The baseline is ``<output>/status_quo.json``. Losses are only comparable when
the dataset fingerprint matches the baseline's — the recency-weighted loss
scales with the number of scored games, so a grown dataset resets the
baseline instead of pretending the numbers compare.

Decision table (strict ``<`` on the recency-weighted best loss):

| status quo | fingerprint matches | loss improves | promote | reason              |
|------------|---------------------|---------------|---------|---------------------|
| absent     | —                   | —             | yes     | initial baseline    |
| present    | yes                 | yes           | yes     | improved            |
| present    | yes                 | no            | no      | no improvement      |
| present    | no                  | n/a           | yes     | baseline reset      |
"""

import dataclasses
import datetime
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from . import constants
from .runs import RunPaths

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    comparable: bool
    baseline_loss: float | None = None


def status_quo_path(output_dir: Path) -> Path:
    return output_dir / constants.STATUS_QUO_FILENAME


def load_status_quo(output_dir: Path) -> dict[str, Any] | None:
    path = status_quo_path(output_dir)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        log.warning("status_quo.json unreadable (%s) — treating as absent", error)
        return None


def decide_promotion(
    status_quo: dict[str, Any] | None,
    fingerprint: str,
    best_loss: float,
) -> PromotionDecision:
    if status_quo is None:
        return PromotionDecision(
            promote=True, reason="initial baseline", comparable=False
        )

    baseline_loss = float(status_quo["metrics"]["best_loss"])
    if status_quo.get("dataset_fingerprint") != fingerprint:
        return PromotionDecision(
            promote=True,
            reason="baseline reset — dataset changed",
            comparable=False,
            baseline_loss=baseline_loss,
        )

    if best_loss < baseline_loss:
        return PromotionDecision(
            promote=True,
            reason=f"improved {baseline_loss:.0f} -> {best_loss:.0f}",
            comparable=True,
            baseline_loss=baseline_loss,
        )
    return PromotionDecision(
        promote=False,
        reason=f"no improvement ({best_loss:.0f} vs {baseline_loss:.0f})",
        comparable=True,
        baseline_loss=baseline_loss,
    )


def apply_promotion(
    paths: RunPaths,
    output_dir: Path,
    run_id: str,
    fingerprint: str,
    metrics: dict[str, float],
) -> None:
    """Deploy the run's model and rewrite the status-quo baseline."""
    shutil.copy2(paths.model_path, output_dir / constants.DEPLOYED_MODEL_FILENAME)
    shutil.copy2(paths.roster_path, output_dir / constants.DEPLOYED_ROSTER_FILENAME)

    model_sha = hashlib.sha256(paths.model_path.read_bytes()).hexdigest()
    baseline = {
        "run_id": run_id,
        "model_sha256": model_sha,
        "metrics": metrics,
        "dataset_fingerprint": fingerprint,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    with open(status_quo_path(output_dir), "w", encoding="utf-8") as handle:
        json.dump(baseline, handle, indent=2)
        handle.write("\n")
    log.info("Promoted %s to deployed model", run_id)
