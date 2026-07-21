# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Pub-quiz score prediction. Game data lives in a Google Sheet
("Inventory - Board Games" → worksheet "Quiz Match History") or a local xlsx
export at `data/board-games.xlsx`. A PyTorch network learns to predict a
team's score from its player composition. Training runs are tracked
ntire-eol-style: one folder per run under `output/runs/`, a newest-first
`output/CHANGELOG.md`, and a promotion gate deciding what `output/model.pt`
(the deployed model) is.

## Commands

`uv` manages the environment (Python ≥ 3.14); [Task](https://taskfile.dev)
wraps the `vopt` CLI:

```bash
task sync                                  # uv sync
task test                                  # pytest — run after any change
task train BEST_OF=100 NOTE="why"          # tracked run -> output/runs/<run_id>/
task changelog                             # newest changelog entries
task preview MODEL=deployed                # games by year (+predictions with MODEL)
task eval MIN_GAMES=3 YEAR=2025            # rank players (variables use UNDERSCORES)
task rank -- --write                       # model-derived strengths -> sheet BE/BF (dry-run without --write)
task anomalies TOP=5 -- --write            # N most extreme outlier flags -> live sheet (dry-run without --write)
task refresh                               # force-refresh snapshot cache
task clean-output                          # DESTRUCTIVE, prompts — wipes runs+changelog
```

Model references: `deployed` (promoted model, default), `latest` (newest
completed run), or an explicit path. Data source: `-- --source {auto,sheets,xlsx,cache}`.

## Architecture (src/variableoptimization/)

Data flows one way: **source → Snapshot → Database → AI → reports**.
Run tracking sits beside it: **cli.run_train → runs.py (layout) + promotion.py (gate)**.

- `snapshot.py` — transport contract (raw strings, row-aligned, worksheet row
  kept for write-backs) + `fingerprint()` (hash of what training sees).
- `sources/` — sheets / xlsx / JSON-cache, interchangeable and identical on
  identical data.
- `database.py` — the **only** parsing site; malformed cells degrade loudly.
- `ai.py` — `ArtificialIntelligence.train()` is pure (no file I/O), returns
  `TrainingResult`; `Predictor` handles all inference; `resolve_model` maps
  deployed/latest/path references.
- `runs.py` — the layout contract: run ids, `RunPaths`, manifests,
  `previous_manifest`, changelog prepend, `latest` symlink.
- `promotion.py` — the status-quo gate (decision table in its docstring).
- `constants.py` — worksheet layout AND output layout; change here, and
  extend the fixture in `tests/conftest.py` to match.

## Invariants — do not break these

- **Never flatten spreadsheet columns.** The Sheets API drops trailing empty
  cells; only interior cells come back as `''`. Always fetch whole rows and
  index columns within each row.
- **Parsing happens once, in `database.py`.**
- **`score is None` means unscored** (blank cells and the sheet's `-1` both
  normalise to it); training uses `Database.scored_games`.
- **Anomaly-flagged games are excluded from training by default**
  (`--include-anomalies` is the escape hatch); the exclusion count is
  recorded in the manifest and changelog. Flagging anomalies changes the
  dataset fingerprint, so the next run legitimately resets the baseline.
- **Constructing a `Game` must not register it with players** — only
  `Database` populates `Player.games`.
- **Models are roster-decoupled.** `roster.json` next to the weights fixes
  the feature-column order forever; `Predictor` must never build features
  from the live database's roster. Unknown players → prediction is `None`,
  never a silent guess.
- **A failed training run must leave no manifest** — manifest, changelog
  entry, and `latest` update happen only after everything else succeeded.
  `previous_manifest`/`find_latest_run` ignore manifest-less dirs by design.
- **Run ids are timestamp-first** (`%Y%m%d_%H%M%S_<sha7>`) so lexical order
  is chronological — previous-run lookup depends on it.
- **Promotion decisions compare losses only within the same dataset
  fingerprint** — the recency-weighted loss scales with the number of scored
  games, so cross-dataset comparisons are meaningless (gate resets instead).
- **Torch imports stay lazy** (`cli.py`, `__init__.py` `__getattr__`).

## Data & secrets

- Google service-account key: `.config/*.json` — gitignored, **never commit
  credentials**. Resolved from `--credentials`, then
  `$GOOGLE_APPLICATION_CREDENTIALS`, then a single `.config/*.json` glob.
- `data/` (xlsx with real people's names) and `output/` (models, changelog)
  are gitignored — keep them that way. Tests use a generated fixture.
- Writes to the live sheet happen only in `SheetsSource.save_anomalies` and
  `SheetsSource.save_rankings`, behind explicit `--write` flags. Keep
  destructive operations opt-in. The rankings block (BE/BF) is write-only —
  never read player strengths back from the sheet; they are derived from the
  model by `vopt rank`.

## Testing

`task test` (pytest, also run in CI on push/PR to master). The fixture in
`tests/conftest.py` generates a small xlsx on the fly containing every edge
case the live sheet has produced: mid-column blank scores, `-1` scores,
`N/A` players, overnight durations.
When you fix a data bug, add its shape to the fixture first. Run-tracking
tests use `tmp_path` and monkeypatch `constants.OUTPUT_DIR` — never write to
the real `output/` from tests.
