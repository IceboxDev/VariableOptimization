"""Run tracking: id contract, layout, previous-run lookup, changelog, latest."""

import datetime

import pytest

from variableoptimization import runs
from variableoptimization.runs import (
    RUN_ID_PATTERN,
    RunPaths,
    create_run_dir,
    find_latest_run,
    format_changelog_entry,
    generate_run_id,
    prepend_changelog,
    previous_manifest,
    update_latest,
    write_manifest,
)


def fixed_time(hour: int = 12) -> datetime.datetime:
    return datetime.datetime(2026, 7, 21, hour, 0, 0, tzinfo=datetime.UTC)


def manifest_stub(run_id: str, best_loss: float = 100.0, **overrides) -> dict:
    manifest = {
        "run_id": run_id,
        "created_at": "2026-07-21T12:00:00+00:00",
        "git": {"sha": "abc123def", "dirty": False},
        "note": "test",
        "dataset": {
            "fingerprint": "f" * 16, "games": 6, "scored_games": 3, "players": 4,
        },
        "config": {"best_of": 2, "seed": None, "workers": 1, "epochs": 5},
        "metrics": {"best_loss": best_loss, "mean_loss": 110.0, "std_loss": 5.0},
        "delta_vs_prev": None,
        "promoted": True,
        "promotion_reason": "initial baseline",
    }
    manifest.update(overrides)
    return manifest


class TestRunId:
    def test_deterministic_format(self, monkeypatch):
        monkeypatch.setattr(runs, "git_state", lambda cwd=None: ("ab12cd3", False))
        assert generate_run_id(now=fixed_time()) == "20260721_120000_ab12cd3"

    def test_dirty_and_suffix_segments(self, monkeypatch):
        monkeypatch.setattr(runs, "git_state", lambda cwd=None: ("ab12cd3", True))
        run_id = generate_run_id(now=fixed_time(), suffix="demo")
        assert run_id == "20260721_120000_ab12cd3_dirty_demo"
        assert RUN_ID_PATTERN.match(run_id)

    def test_nogit_fallback_is_valid(self, monkeypatch):
        monkeypatch.setattr(runs, "git_state", lambda cwd=None: ("nogit", False))
        assert RUN_ID_PATTERN.match(generate_run_id(now=fixed_time()))

    def test_lexical_order_is_chronological(self, monkeypatch):
        monkeypatch.setattr(runs, "git_state", lambda cwd=None: ("ab12cd3", False))
        earlier = generate_run_id(now=fixed_time(hour=9))
        later = generate_run_id(now=fixed_time(hour=15))
        assert sorted([later, earlier]) == [earlier, later]

    def test_real_git_state_matches_pattern(self):
        assert RUN_ID_PATTERN.match(generate_run_id())


class TestRunDirs:
    def test_create_and_refuse_overwrite(self, tmp_path):
        paths = create_run_dir(tmp_path, "20260721_120000_ab12cd3")
        assert paths.root.is_dir()
        with pytest.raises(FileExistsError):
            create_run_dir(tmp_path, "20260721_120000_ab12cd3")

    def test_previous_manifest_skips_failed_runs(self, tmp_path):
        first = create_run_dir(tmp_path, "20260721_090000_aaaaaaa")
        write_manifest(first, manifest_stub(first.root.name, best_loss=90.0))
        # A failed run: directory exists, no manifest.
        create_run_dir(tmp_path, "20260721_100000_bbbbbbb")

        previous = previous_manifest(tmp_path, "20260721_120000_ccccccc")
        assert previous is not None
        assert previous["run_id"] == "20260721_090000_aaaaaaa"

    def test_previous_manifest_none_for_first_run(self, tmp_path):
        assert previous_manifest(tmp_path, "20260721_120000_ab12cd3") is None

    def test_find_latest_ignores_manifestless_dirs(self, tmp_path):
        assert find_latest_run(tmp_path) is None
        completed = create_run_dir(tmp_path, "20260721_090000_aaaaaaa")
        write_manifest(completed, manifest_stub(completed.root.name))
        create_run_dir(tmp_path, "20260721_100000_bbbbbbb")  # failed, no manifest

        assert find_latest_run(tmp_path) == completed.root


class TestLatestSymlink:
    def test_points_at_run_and_repoints(self, tmp_path):
        create_run_dir(tmp_path, "20260721_090000_aaaaaaa")
        create_run_dir(tmp_path, "20260721_100000_bbbbbbb")

        update_latest(tmp_path, "20260721_090000_aaaaaaa")
        link = tmp_path / "latest"
        assert link.resolve().name == "20260721_090000_aaaaaaa"

        update_latest(tmp_path, "20260721_100000_bbbbbbb")
        assert link.resolve().name == "20260721_100000_bbbbbbb"


class TestChangelog:
    def test_first_entry_creates_header(self, tmp_path):
        path = tmp_path / "CHANGELOG.md"
        prepend_changelog(path, "## run-a\n\nentry a\n\n")
        content = path.read_text()
        assert content.startswith("# Training Changelog")
        assert "## run-a" in content

    def test_prepend_is_newest_first(self, tmp_path):
        path = tmp_path / "CHANGELOG.md"
        prepend_changelog(path, "## run-a\n\nentry a\n\n")
        prepend_changelog(path, "## run-b\n\nentry b\n\n")
        content = path.read_text()
        assert content.index("## run-b") < content.index("## run-a")
        assert content.count("# Training Changelog") == 1

    def test_entry_formatting(self):
        manifest = manifest_stub(
            "20260721_120000_ab12cd3",
            best_loss=95.0,
            delta_vs_prev={
                "prev_run_id": "20260721_090000_aaaaaaa",
                "best_loss": -5.0,
                "comparable": True,
            },
            promoted=True,
            promotion_reason="improved 100 -> 95",
        )
        entry = format_changelog_entry(manifest)
        assert "## 20260721_120000_ab12cd3" in entry
        assert "✅ promoted — improved 100 -> 95" in entry
        assert "-5 vs 20260721_090000_aaaaaaa" in entry
        assert "runs/20260721_120000_ab12cd3/" in entry

    def test_entry_formatting_not_comparable(self):
        manifest = manifest_stub(
            "20260721_120000_ab12cd3",
            delta_vs_prev={
                "prev_run_id": "20260721_090000_aaaaaaa",
                "best_loss": None,
                "comparable": False,
            },
            promoted=False,
            promotion_reason="no improvement (100 vs 90)",
        )
        entry = format_changelog_entry(manifest)
        assert "❌ not promoted" in entry
        assert "not comparable" in entry
