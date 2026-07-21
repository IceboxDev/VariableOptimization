# VariableOptimization

Predicts pub-quiz scores from team composition. Game data lives in a Google
Sheet ("Inventory - Board Games" → "Quiz Match History"); a PyTorch network is
trained on historical games and used to preview predictions, rank players, and
flag anomalous results. Every training run is tracked: one folder per run, a
changelog, and a promotion gate deciding which model is deployed.

## Setup

```bash
uv sync                       # or: task sync
```

Place a Google service-account key at `.config/<key>.json` (gitignored). It is
found automatically; alternatively set `GOOGLE_APPLICATION_CREDENTIALS` or
pass `--credentials`.

## Usage

Everything runs through [Task](https://taskfile.dev):

```bash
task train BEST_OF=100 NOTE="why this run"   # tracked run (anomaly-flagged games excluded)
task preview MODEL=deployed                  # games by year, with predictions
task eval MIN_GAMES=3 YEAR=2025              # rank players by predicted team scores
task rank                                    # per-player strength from the model (dry run)
task rank -- --write                         # ...and write it to the sheet's BE/BF block
task anomalies                               # outlier report (dry run)
task anomalies TOP=5 -- --write              # write the 5 most extreme flags to the sheet
```

`MODEL=` accepts `deployed` (the promoted model — default), `latest` (newest
completed run), or an explicit path. To bypass a fresh-but-stale cache (e.g.
right after `--write`-ing flags), append `-- --refresh` to any command.

Developer commands run directly: `uv sync`, `uv run pytest`,
`uv run vopt --help`.

## Run tracking

Each `task train` creates `output/runs/<UTC timestamp>_<git sha7>[_dirty][_<suffix>]/`:

```
model.pt        trained weights
roster.json     player roster the model was trained on (fixes column order)
manifest.json   metrics, dataset fingerprint, delta vs previous run, promotion verdict
config.json     training parameters
loss.png        loss distribution across the best-of candidates
train.log       full training log
```

`output/CHANGELOG.md` (newest first) records every run: note, ✅/❌ promotion
status with reason, loss delta versus the previous run, git and dataset
inputs. `output/runs/latest` always points at the newest successful run.

**Promotion gate** — a run only becomes the deployed model (`output/model.pt`)
if it beats the baseline in `output/status_quo.json`:

| Baseline | Dataset fingerprint | Loss | Verdict |
|---|---|---|---|
| none yet | — | — | ✅ initial baseline |
| exists | same | improved | ✅ promoted |
| exists | same | worse/equal | ❌ not promoted |
| exists | changed | (not comparable) | ✅ baseline reset — dataset changed |

Models are roster-decoupled: the trained roster ships with the weights, so a
model keeps working after new players join — games involving players the model
never saw show no prediction instead of a wrong one.

## Data sources

Every command accepts `--source` (via `task <cmd> -- --source xlsx`):

| Source   | Behaviour                                                                  |
|----------|----------------------------------------------------------------------------|
| `auto`   | Default. Cache if fresher than 3 h, else Google Sheets (falls back to a stale cache if Sheets is unreachable). |
| `sheets` | Always fetch from Google Sheets and refresh the cache.                     |
| `xlsx`   | Read a local export of the spreadsheet (`data/board-games.xlsx`). Fully offline. |
| `cache`  | Use `.cache/snapshot.json` regardless of age. Fully offline.               |

All sources produce the same snapshot format, so results are identical for
identical data (verified by the test suite).

## Architecture

```
src/variableoptimization/
├── constants.py     # worksheet layout, game rules, training config, output layout
├── domain.py        # Player / Game dataclasses
├── snapshot.py      # transport contract all sources produce + dataset fingerprint
├── sources/
│   ├── sheets.py    # Google Sheets (row-aligned fetch, anomaly write-back)
│   ├── xlsx.py      # local spreadsheet clone
│   └── cache.py     # versioned JSON cache with TTL
├── loader.py        # source selection / fallback logic
├── database.py      # snapshot -> domain parsing (the only parsing site)
├── ai.py            # NeuralNetwork, training, Predictor (roster-decoupled inference)
├── runs.py          # run tracking: run ids, manifests, changelog, latest symlink
├── promotion.py     # status-quo promotion gate
├── reports.py       # preview tables, player ranking, anomaly detection
└── cli.py           # `vopt` entry point (wrapped by Taskfile)
```

CI runs the test suite on every push/PR to master (`.github/workflows/ci.yml`).
