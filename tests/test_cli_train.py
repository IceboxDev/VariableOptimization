"""End-to-end: the tracked-run train pipeline through the real CLI."""

import json

import pytest

from variableoptimization import constants
from variableoptimization.cli import main


@pytest.fixture()
def output_dir(tmp_path, monkeypatch):
    out = tmp_path / "output"
    monkeypatch.setattr(constants, "OUTPUT_DIR", str(out))
    return out


def train(fixture_xlsx, extra: list[str] | None = None) -> int:
    return main([
        "--source", "xlsx", "--xlsx", str(fixture_xlsx),
        "train", "--best-of", "1", "--epochs", "5", "--seed", "7",
        *(extra or []),
    ])


def test_first_run_seeds_the_full_structure(fixture_xlsx, output_dir):
    assert train(fixture_xlsx, ["--note", "first"]) == 0

    runs_root = output_dir / "runs"
    run_dirs = [d for d in runs_root.iterdir() if d.is_dir() and not d.is_symlink()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    for artifact in (
        "model.pt", "roster.json", "manifest.json", "config.json",
        "loss.png", "train.log",
    ):
        assert (run_dir / artifact).is_file(), f"missing {artifact}"

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["promoted"] is True
    assert manifest["promotion_reason"] == "initial baseline"
    assert manifest["delta_vs_prev"] is None
    assert manifest["note"] == "first"
    assert manifest["dataset"]["players"] == 4

    assert (runs_root / "latest").resolve() == run_dir
    assert (output_dir / "model.pt").is_file()
    assert (output_dir / "roster.json").is_file()
    assert (output_dir / "status_quo.json").is_file()

    changelog = (output_dir / "CHANGELOG.md").read_text()
    assert changelog.startswith("# Training Changelog")
    assert "✅ promoted — initial baseline" in changelog


def test_second_run_populates_delta_and_gate(fixture_xlsx, output_dir):
    assert train(fixture_xlsx, ["--note", "first"]) == 0
    assert train(fixture_xlsx, ["--note", "second", "--suffix", "again"]) == 0

    runs_root = output_dir / "runs"
    run_dirs = sorted(
        d for d in runs_root.iterdir() if d.is_dir() and not d.is_symlink()
    )
    assert len(run_dirs) == 2
    second = json.loads((run_dirs[-1] / "manifest.json").read_text())

    assert second["delta_vs_prev"] is not None
    assert second["delta_vs_prev"]["prev_run_id"] == run_dirs[0].name
    assert second["delta_vs_prev"]["comparable"] is True  # same fixture data
    reason = second["promotion_reason"]
    assert ("improved" in reason) or ("no improvement" in reason)

    # latest tracks the newest successful run regardless of promotion.
    assert (runs_root / "latest").resolve() == run_dirs[-1]

    changelog = (output_dir / "CHANGELOG.md").read_text()
    assert changelog.index("**Note:** second") < changelog.index("**Note:** first")
    assert changelog.count("# Training Changelog") == 1


def test_eval_works_off_the_deployed_model(fixture_xlsx, output_dir, capsys):
    assert train(fixture_xlsx) == 0
    code = main([
        "--source", "xlsx", "--xlsx", str(fixture_xlsx),
        "eval", "--min-games", "1",
    ])
    assert code == 0
    # 4 players < TEAM_SIZE(5): eval reports the too-few-players message,
    # which proves deployed-model resolution + Predictor.load succeeded.
    assert "match the filters" in capsys.readouterr().out


def test_failed_run_leaves_no_manifest_and_no_latest(fixture_xlsx, output_dir, monkeypatch):
    from variableoptimization import ai

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ai.ArtificialIntelligence, "train", explode)
    assert train(fixture_xlsx) == 1

    runs_root = output_dir / "runs"
    run_dirs = [d for d in runs_root.iterdir() if d.is_dir() and not d.is_symlink()]
    assert len(run_dirs) == 1
    assert not (run_dirs[0] / "manifest.json").exists()
    assert (run_dirs[0] / "train.log").is_file()
    assert not (runs_root / "latest").exists()
    assert not (output_dir / "CHANGELOG.md").exists()
