# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Pub-quiz score prediction. Game data lives in a Google Sheet
("Inventory - Board Games" → worksheet "Quiz Match History") or a local xlsx
export of it. A PyTorch network learns to predict a team's score from its
player composition; the CLI previews games, ranks players, and flags
anomalous results.

## Commands

`uv` manages the environment (Python ≥ 3.14); [Task](https://taskfile.dev)
wraps the `vopt` CLI:

```bash
task sync                          # uv sync
task test                          # pytest — run after any change
task train BEST_OF=100 SEED=42     # train, keep best of N candidates
task preview MODEL=latest          # games by year (+predictions with MODEL)
task eval MIN_GAMES=3 YEAR=2025    # rank players by predicted team scores
task anomalies                     # outlier report, dry-run
task anomalies -- --write          # ...and write flags to the live sheet
task refresh                       # force-refresh the snapshot cache
```

`uv run vopt --help` for the full CLI. Every command accepts
`--source {auto,sheets,xlsx,cache}` — use `--source xlsx` or `--source cache`
to work offline; `auto` prefers a <3 h cache, then Sheets, then a stale cache.

## Architecture (src/variableoptimization/)

Data flows one way: **source → Snapshot → Database → AI → reports**.

- `snapshot.py` — the transport contract. Every source produces the same
  `Snapshot`: raw *strings* in worksheet conventions (`MM/DD/YY` dates,
  `TRUE`/`FALSE` anomalies), row-aligned per record, with the originating
  worksheet row kept for write-backs.
- `sources/` — `sheets.py` (live sheet), `xlsx.py` (offline clone),
  `cache.py` (versioned JSON, 3 h TTL, atomic writes). Interchangeable and
  verified identical on identical data.
- `database.py` — the **only** place snapshot strings are parsed into domain
  objects. Malformed cells degrade with a logged warning, never crash.
- `domain.py` — `Player` (identity = name) and `Game` (frozen) dataclasses.
- `loader.py` — `DataSettings` → source selection/fallback → `Database`.
- `ai.py` — `NeuralNetwork`, training orchestration, model persistence.
- `reports.py` — rich tables; `cli.py` — argparse entry point (`vopt`).
- `constants.py` — worksheet layout (1-based rows/columns), game rules,
  training config. When the spreadsheet structure changes, change it here
  and extend the test fixture to match.

## Invariants — do not break these

- **Never flatten spreadsheet columns.** The Sheets API drops trailing empty
  cells; only interior cells come back as `''`. Always fetch whole rows and
  index columns within each row, so a blank cell can't shift later values
  into the wrong game. This bug class corrupted data silently before.
- **Parsing happens once, in `database.py`.** Sources canonicalise values to
  sheet string conventions; nothing downstream touches raw cells.
- **`score is None` means unscored.** Blank cells and the sheet's `-1`
  sentinel both normalise to `None`; training uses `Database.scored_games`.
- **Constructing a `Game` must not register it with players.** Only
  `Database` populates `Player.games` — hypothetical games built for
  inference must leave histories untouched.
- **Player column order is sorted-by-name** and fixes the model's feature
  vector. Changing it silently invalidates every saved model.
- **Torch imports stay lazy** (`cli.py`, `__init__.py` `__getattr__`) so
  data-only commands start in ~0.2 s.

## Data & secrets

- Google service-account key: `.config/*.json` — gitignored, **never commit
  credentials**. Resolved from `--credentials`, then
  `$GOOGLE_APPLICATION_CREDENTIALS`, then a single `.config/*.json` glob.
- `Inventory - Board Games.xlsx` contains real people's names — gitignored,
  keep it that way. Tests use a sanitized generated fixture instead.
- Writes to the live sheet happen only in `SheetsSource.save_anomalies`,
  behind the explicit `--write` flag. Keep destructive operations opt-in.

## Testing

`task test`. The fixture in `tests/conftest.py` generates a small xlsx on the
fly containing every edge case the live sheet has produced: mid-column blank
scores, `-1` scores, an Overlap weight row with an empty cell, `N/A` players,
overnight durations. When you fix a data bug, add its shape to the fixture
first so the fix is pinned by a test.

## Models

Saved to `models/` (gitignored) as `neuralnetwork-<loss>.pt` with a
loss-distribution plot alongside; `--model latest` resolves the newest by
mtime. Models are tied to the player-roster size they were trained with —
after new players join, old models fail to load with a clear message and a
retrain is required. Reproducibility: `--seed` with `--workers 1` (default).
