"""Source resolution: turns CLI settings into a Database.

Modes:
- ``auto``   — fresh cache if available, otherwise Google Sheets (refreshing
               the cache). If Sheets is unreachable, falls back to a stale
               cache with a warning rather than dying.
- ``sheets`` — always fetch from Google Sheets (and refresh the cache).
- ``xlsx``   — read the local xlsx clone; never touches the network.
- ``cache``  — use the cache regardless of age; fail if there is none.
"""

import dataclasses
import glob
import logging
import os
from pathlib import Path

from . import constants
from .database import Database
from .snapshot import Snapshot
from .sources import SheetsSource, SnapshotCache, XlsxSource

log = logging.getLogger(__name__)


class DataSourceError(Exception):
    """A data source could not be resolved or read."""


@dataclasses.dataclass
class DataSettings:
    source: str = "auto"
    xlsx_path: Path = Path(constants.XLSX_DEFAULT_PATH)
    credentials: Path | None = None
    cache_path: Path = Path(constants.CACHE_DEFAULT_PATH)
    refresh: bool = False


def resolve_credentials(explicit: Path | None = None) -> Path:
    """Find the service-account key: explicit flag > env var > .config glob."""
    if explicit is not None:
        if not explicit.is_file():
            raise DataSourceError(f"Credentials file not found: {explicit}")
        return explicit

    from_env = os.environ.get(constants.CREDENTIALS_ENV_VAR)
    if from_env:
        path = Path(from_env)
        if not path.is_file():
            raise DataSourceError(
                f"{constants.CREDENTIALS_ENV_VAR} points to a missing file: {path}"
            )
        return path

    matches = glob.glob(constants.CREDENTIALS_GLOB)
    if len(matches) == 1:
        return Path(matches[0])
    raise DataSourceError(
        "No Google credentials found. Pass --credentials, set "
        f"{constants.CREDENTIALS_ENV_VAR}, or place exactly one key at "
        f"{constants.CREDENTIALS_GLOB} (found {len(matches)})."
    )


def _fetch_and_cache(settings: DataSettings, cache: SnapshotCache) -> Snapshot:
    source = SheetsSource(resolve_credentials(settings.credentials))
    snapshot = source.fetch()
    cache.save(snapshot)
    return snapshot


def load_snapshot(settings: DataSettings) -> Snapshot:
    cache = SnapshotCache(settings.cache_path)

    match settings.source:
        case "xlsx":
            return XlsxSource(settings.xlsx_path).fetch()

        case "cache":
            snapshot = cache.load(max_age=None)
            if snapshot is None:
                raise DataSourceError(
                    f"No usable cache at {settings.cache_path} — "
                    "run with --source sheets (or 'vopt refresh') first."
                )
            return snapshot

        case "sheets":
            return _fetch_and_cache(settings, cache)

        case "auto":
            if not settings.refresh:
                snapshot = cache.load()
                if snapshot is not None:
                    log.debug("Using cached snapshot (%s)", settings.cache_path)
                    return snapshot
            try:
                return _fetch_and_cache(settings, cache)
            except Exception as error:  # auth/transport errors vary by layer
                stale = cache.load(max_age=None)
                if stale is not None:
                    log.warning(
                        "Google Sheets unavailable (%s) — using stale cache",
                        error,
                    )
                    return stale
                raise

        case other:
            raise DataSourceError(f"Unknown source {other!r}")


def load_database(settings: DataSettings | None = None) -> Database:
    return Database(load_snapshot(settings or DataSettings()))
