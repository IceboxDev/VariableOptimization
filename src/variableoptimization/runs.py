"""Run tracking: the layout contract for training outputs.

One directory per training run under ``<output>/runs/<run_id>/``, holding the
model, its roster, a manifest with metrics and a machine-readable delta versus
the previous run, the config snapshot, the loss plot, and the training log.
``CHANGELOG.md`` lives one level above ``runs/`` and is prepended newest-first.
``runs/latest`` is a symlink updated only after a fully successful run —
failed runs keep their log but never gain a manifest, which makes them
invisible to ``latest``, the changelog, and previous-run lookups.

Run ids are ``<UTC %Y%m%d_%H%M%S>_<git sha7>[_dirty][_<suffix>]``; the
timestamp prefix makes lexical order equal chronological order, which the
previous-run lookup relies on.
"""

import dataclasses
import datetime
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Final

from . import constants

log = logging.getLogger(__name__)

RUN_ID_PATTERN: Final = re.compile(
    r"^\d{8}_\d{6}_[0-9a-f]{7}(?:_dirty)?(?:_[A-Za-z0-9-]+)?$|"
    r"^\d{8}_\d{6}_nogit(?:_dirty)?(?:_[A-Za-z0-9-]+)?$"
)

CHANGELOG_HEADER: Final = (
    "# Training Changelog\n"
    "\n"
    "Newest first. Each entry records one training run; artifacts live in\n"
    "`runs/<run_id>/`. The deployed model is `model.pt` next to this file —\n"
    "promotion is gated on `status_quo.json` (see manifest for details).\n"
)


def git_state(cwd: Path | None = None) -> tuple[str, bool]:
    """Return (sha7, dirty) for the repository at ``cwd``.

    Falls back to ("nogit", False) when git or a repository is unavailable —
    a missing repo must never block training.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd, capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "nogit", False


def generate_run_id(
    now: datetime.datetime | None = None,
    suffix: str | None = None,
    cwd: Path | None = None,
) -> str:
    if now is None:
        now = datetime.datetime.now(datetime.UTC)
    sha, dirty = git_state(cwd)
    run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{sha}"
    if dirty:
        run_id += "_dirty"
    if suffix:
        run_id += f"_{suffix}"
    if not RUN_ID_PATTERN.match(run_id):
        raise ValueError(f"Generated run id {run_id!r} violates the id contract")
    return run_id


@dataclasses.dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def model_path(self) -> Path:
        return self.root / "model.pt"

    @property
    def roster_path(self) -> Path:
        return self.root / "roster.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def plot_path(self) -> Path:
        return self.root / "loss.png"

    @property
    def log_path(self) -> Path:
        return self.root / "train.log"


def runs_root(output_dir: Path) -> Path:
    return output_dir / constants.RUNS_DIRNAME


def changelog_path(output_dir: Path) -> Path:
    return output_dir / constants.CHANGELOG_FILENAME


def create_run_dir(root: Path, run_id: str) -> RunPaths:
    """Create ``<root>/<run_id>``; refuse to reuse an existing run directory."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunPaths(run_dir)


def write_manifest(paths: RunPaths, manifest: dict[str, Any]) -> None:
    with open(paths.manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def read_manifest(run_dir: Path) -> dict[str, Any]:
    with open(run_dir / "manifest.json", encoding="utf-8") as handle:
        return json.load(handle)


def _completed_runs(root: Path) -> list[Path]:
    """Run directories that finished (i.e. have a manifest), lexically sorted."""
    if not root.is_dir():
        return []
    return sorted(
        entry
        for entry in root.iterdir()
        if entry.is_dir()
        and not entry.is_symlink()
        and (entry / "manifest.json").is_file()
    )


def previous_manifest(root: Path, run_id: str) -> dict[str, Any] | None:
    """Manifest of the newest completed run older than ``run_id``."""
    older = [run for run in _completed_runs(root) if run.name < run_id]
    if not older:
        return None
    return read_manifest(older[-1])


def find_latest_run(root: Path) -> Path | None:
    completed = _completed_runs(root)
    return completed[-1] if completed else None


def update_latest(root: Path, run_id: str) -> None:
    """Atomically point ``<root>/latest`` at ``run_id``."""
    link = root / constants.LATEST_LINK_NAME
    temporary = root / f".{constants.LATEST_LINK_NAME}.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(run_id)
    os.replace(temporary, link)


def prepend_changelog(path: Path, entry: str) -> None:
    """Insert ``entry`` directly under the fixed header, newest-first."""
    if path.is_file():
        tail = path.read_text(encoding="utf-8").removeprefix(CHANGELOG_HEADER)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tail = ""
    path.write_text(CHANGELOG_HEADER + "\n" + entry + tail, encoding="utf-8")


def format_changelog_entry(manifest: dict[str, Any]) -> str:
    run_id = manifest["run_id"]
    note = manifest.get("note") or "(no note)"
    stamp = "✅ promoted" if manifest["promoted"] else "❌ not promoted"
    metrics = manifest["metrics"]

    delta = manifest.get("delta_vs_prev")
    if delta is None:
        loss_line = f"- best_loss: {metrics['best_loss']:.0f} (first run)"
    elif delta["comparable"]:
        loss_line = (
            f"- best_loss: {metrics['best_loss']:.0f} "
            f"({delta['best_loss']:+.0f} vs {delta['prev_run_id']})"
        )
    else:
        loss_line = (
            f"- best_loss: {metrics['best_loss']:.0f} "
            f"(dataset changed since {delta['prev_run_id']} — not comparable)"
        )

    dataset = manifest["dataset"]
    return (
        f"## {run_id}\n"
        f"\n"
        f"**Note:** {note}\n"
        f"\n"
        f"**Status:** {stamp} — {manifest['promotion_reason']}\n"
        f"\n"
        f"**Metrics**\n"
        f"{loss_line}\n"
        f"- mean ± std over {manifest['config']['best_of']} candidates: "
        f"{metrics['mean_loss']:.0f} ± {metrics['std_loss']:.0f}\n"
        f"\n"
        f"**Inputs**\n"
        f"- git: `{manifest['git']['sha']}`"
        f"{' (dirty)' if manifest['git']['dirty'] else ''}\n"
        f"- dataset: `{dataset['fingerprint']}` — "
        f"{dataset['scored_games']} scored games, {dataset['players']} players\n"
        f"\n"
        f"Artifacts: [`runs/{run_id}/`](runs/{run_id}/)\n"
        f"\n"
    )
