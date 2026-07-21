"""Cache: round-trip fidelity, TTL, schema versioning, corruption handling."""

import datetime
import json

from variableoptimization.sources import SnapshotCache


def test_round_trip(tmp_path, fixture_snapshot):
    cache = SnapshotCache(tmp_path / "snapshot.json")
    cache.save(fixture_snapshot)
    assert cache.load() == fixture_snapshot


def test_missing_file_returns_none(tmp_path):
    assert SnapshotCache(tmp_path / "nope.json").load() is None


def test_expired_cache_returns_none(tmp_path, fixture_snapshot):
    path = tmp_path / "snapshot.json"
    cache = SnapshotCache(path, ttl=datetime.timedelta(hours=3))
    cache.save(fixture_snapshot)

    payload = json.loads(path.read_text())
    payload["timestamp"] -= 4 * 3600
    path.write_text(json.dumps(payload))

    assert cache.load() is None
    # max_age=None accepts any age — used for the stale-cache fallback.
    assert cache.load(max_age=None) == fixture_snapshot


def test_schema_mismatch_returns_none(tmp_path, fixture_snapshot, caplog):
    path = tmp_path / "snapshot.json"
    cache = SnapshotCache(path)
    cache.save(fixture_snapshot)

    payload = json.loads(path.read_text())
    payload["schema_version"] = 1
    path.write_text(json.dumps(payload))

    with caplog.at_level("WARNING"):
        assert cache.load() is None
    assert any("schema" in message for message in caplog.messages)


def test_corrupt_file_returns_none(tmp_path, caplog):
    path = tmp_path / "snapshot.json"
    path.write_text("{ not json")
    with caplog.at_level("WARNING"):
        assert SnapshotCache(path).load() is None
    assert any("unreadable" in message for message in caplog.messages)


def test_save_creates_parent_directory(tmp_path, fixture_snapshot):
    cache = SnapshotCache(tmp_path / "nested" / "dir" / "snapshot.json")
    cache.save(fixture_snapshot)
    assert cache.load() == fixture_snapshot
