# VariableOptimization

Predicts pub-quiz scores from team composition. Game data lives in a Google
Sheet ("Inventory - Board Games" → "Quiz Match History"); a PyTorch network is
trained on historical games and used to preview predictions, rank players, and
flag anomalous results.

## Setup

```bash
uv sync                       # or: task sync
```

Place a Google service-account key at `.config/<key>.json` (gitignored). It is
found automatically; alternatively set `GOOGLE_APPLICATION_CREDENTIALS` or
pass `--credentials`.

## Usage

Everything runs through [Task](https://taskfile.dev) or the `vopt` CLI directly:

```bash
task train BEST_OF=100 SEED=42    # train, keep best of N candidates
task preview MODEL=latest         # games by year, with predictions
task eval MIN_GAMES=3 YEAR=2025   # rank players by predicted team scores
task anomalies                    # outlier report (dry run)
task anomalies -- --write         # ...and write flags back to the sheet
task refresh                      # force-refresh the snapshot cache
task test                         # run the test suite
```

`vopt --help` shows the full CLI, including data-source selection.

## Data sources

Every command accepts `--source`:

| Source   | Behaviour                                                                  |
|----------|----------------------------------------------------------------------------|
| `auto`   | Default. Cache if fresher than 3 h, else Google Sheets (falls back to a stale cache if Sheets is unreachable). |
| `sheets` | Always fetch from Google Sheets and refresh the cache.                     |
| `xlsx`   | Read a local export of the spreadsheet (`--xlsx`, defaults to `Inventory - Board Games.xlsx`). Fully offline. |
| `cache`  | Use `.cache/snapshot.json` regardless of age. Fully offline.               |

All sources produce the same snapshot format, so results are identical for
identical data (verified by the test suite).

## Architecture

```
src/variableoptimization/
├── constants.py     # worksheet layout, game rules, training config
├── domain.py        # Player / Game dataclasses
├── snapshot.py      # transport contract all sources produce
├── sources/
│   ├── sheets.py    # Google Sheets (row-aligned fetch, anomaly write-back)
│   ├── xlsx.py      # local spreadsheet clone
│   └── cache.py     # versioned JSON cache with TTL
├── loader.py        # source selection / fallback logic
├── database.py      # snapshot -> domain parsing (the only parsing site)
├── ai.py            # NeuralNetwork + training/inference orchestration
├── reports.py       # preview tables, player ranking, anomaly detection
└── cli.py           # `vopt` entry point
```

Trained models land in `models/` (gitignored) as `neuralnetwork-<loss>.pt`
with a loss-distribution plot alongside. `--model latest` picks the newest.
