"""Snapshot cache: versioned JSON with a TTL, written atomically.

Replaces the old unversioned pickle cache — human-inspectable, safe to load,
and rejected loudly (with a log line) instead of silently when invalid.
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path

from .. import constants
from ..snapshot import SCHEMA_VERSION, Snapshot

log = logging.getLogger(__name__)

_DEFAULT_TTL = object()  # sentinel: "use the configured TTL"


class SnapshotCache:
    def __init__(
        self,
        path: str | Path = constants.CACHE_DEFAULT_PATH,
        ttl: datetime.timedelta = constants.CACHE_TTL,
    ) -> None:
        self._path = Path(path)
        self._ttl = ttl

    def load(self, max_age: datetime.timedelta | None | object = _DEFAULT_TTL) -> Snapshot | None:
        """Return the cached snapshot, or None if missing, stale, or invalid.

        ``max_age=None`` disables the age check (accept any cache).
        """
        if max_age is _DEFAULT_TTL:
            max_age = self._ttl

        if not self._path.is_file():
            return None

        try:
            with open(self._path, encoding="utf-8") as handle:
                payload = json.load(handle)

            if payload.get("schema_version") != SCHEMA_VERSION:
                log.warning(
                    "Cache %s has schema %r, expected %r — ignoring",
                    self._path, payload.get("schema_version"), SCHEMA_VERSION,
                )
                return None

            age = time.time() - float(payload["timestamp"])
            if max_age is not None and age > max_age.total_seconds():
                log.debug("Cache %s is stale (%.0f s old)", self._path, age)
                return None

            return Snapshot.from_dict(payload["snapshot"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            log.warning("Cache %s is unreadable (%s) — ignoring", self._path, error)
            return None

    def save(self, snapshot: Snapshot) -> None:
        """Write atomically (tmp + rename). Failure is logged, not raised —
        a broken cache must never take the pipeline down."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": time.time(),
            "snapshot": snapshot.to_dict(),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(".tmp")
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temporary, self._path)
        except OSError as error:
            log.warning("Could not write cache %s: %s", self._path, error)

    def age(self) -> datetime.timedelta | None:
        """Age of the cache file, or None if there is none."""
        try:
            with open(self._path, encoding="utf-8") as handle:
                timestamp = float(json.load(handle)["timestamp"])
        except (OSError, ValueError, KeyError):
            return None
        return datetime.timedelta(seconds=time.time() - timestamp)
