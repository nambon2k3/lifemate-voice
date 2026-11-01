# Project structure

LifeMate turns one spoken Vietnamese request into an ordered list of tool calls and a final `state_hash`. The official data kit is the source of samples, tool schemas, and the simulator. Code we write only reads that kit and writes a `submission.jsonl`.

```text
lifemate-voice/
├── README.md                          challenge brief
├── STRUCTURE.md                       this file
├── CONTEXT.MD                         local notes
│
├── train+public_test/                 official kit v3.2.0 (read-only)
│   ├── README.md                      kit format, ToolEnv, submission rules
│   ├── VERSION                        release, build time, sample counts
│   ├── train/                         1545 labeled samples
│   └── public_test/                   634 unlabeled samples
│
├── src/                               agent code
│   ├── asr.py                         speech to text, cached
│   ├── context.py                     time, place, profile, history as text
│   ├── agent.py                       one tool call per step
│   ├── validate.py                    reject a call before env.call
│   └── run.py                         one sample, or a full split, to JSONL
│
├── configs/
│   └── agent.yaml                     model ids, temperature, step cap, paths
├── scripts/
│   └── score_dev.py                   holdout score against train labels
├── cache/
│   └── transcripts/                   PhoWhisper output, keyed by sample id
├── outputs/                           generated submission.jsonl and run logs
└── models/                            local ASR and LLM weights
```

`train+public_test/` is already in the repo (1545 train, 634 public test, kit `v3.2.0`). `src/`, `configs/`, `scripts/`, `cache/`, `outputs/`, and `models/` are the code layout. Leave the kit unchanged: the organizer scores by replaying `tool_calls` with the original `zaic_tools` and the original `state.json`.

## Official kit

Path: `train+public_test/`. Full field definitions are in `train+public_test/README.md`.

Each split is self-contained. Train and public test ship separate catalogs and separate `zaic_tools` copies. A private test split is released later and follows the same shape.

```text
train/   or   public_test/
├── <split>.jsonl
├── <split>_tools_catalog.json
├── <split>_environment/<id>/state.json
├── audio/<id>/
│   ├── <id>.wav              latest user request (this is what is scored)
│   └── <id>_u1.wav ...       earlier user turns in history
├── sample_submission.jsonl   public test only, three format examples
└── zaic_tools/
```

### `<split>.jsonl`

One JSON object per line. This is the list of tasks.

- `id` — sample id (`train_0001`, `public_0002`). Same id as the environment folder and the submission line.
- `audio` — path of the latest request, relative to the split directory.
- `context` — `current_time`, `timezone`, `location`, `user_profile` (name, home, work, city), and `installed_apps`. A sample may call only tools listed for its installed apps, plus `sys.ask_missing_info`.
- `history` — earlier turns. User turns point at prior wav files. Assistant turns may already contain `tool_calls` and `tool_results`. Those calls are already applied inside `state.json`. Only the latest audio is scored.
- Train only: `audio_text` (transcript of the latest request) and `label`.
- `label.tool_calls` — gold call sequence. `[]` means unsupported or no action.
- `label.tool_results` — result of each gold call, same order.
- `label.final_state` — full state of apps the gold calls changed. Unchanged apps are omitted.

Public test lines have no `audio_text` and no `label`.

### `<split>_tools_catalog.json`

Schema for every app and tool in that split (`schema_version`, `note`, `apps`). Each tool has `name` (`app.tool`), `kind` (`read`, `write`, or `meta`), `description`, JSON Schema `parameters` (`properties`, `required`, and extras such as `x-one-of`, `x-ref`, `x-match`), `returns`, and `app`. The agent must use this file for the current split. Tool names from another split or from memory are invalid.

### `<split>_environment/<id>/`

One folder per sample, containing `state.json`. That file is the user’s app data at the moment of the latest request, including anything history already did. `_meta` stores the question id, current time, and allowed tools. Other keys are app data (`calendar`, `contacts`, `sms`, and the rest).

The agent learns this data by calling tools. `ToolEnv` loads `state.json` into memory. Writes stay in that session. The file on disk stays as shipped.

### `audio/<id>/`

Wav files for that sample. `<id>.wav` is the request to transcribe and act on. `<id>_uN.wav` files are history turns, used when the latest request refers to something said earlier.

### `zaic_tools/`

Local Python tool runtime for that split. The leaderboard replays a submission with this same library. Standard library only, Python 3.10+.

| File | Responsibility |
|---|---|
| `__init__.py` | Exports `ToolEnv`. |
| `__main__.py` | CLI: list apps, call one tool, or schema-check a submission file. |
| `executor.py` | Loads the split, checks schema, runs the app method, records every call. |
| `validate.py` | The only reject rules: unknown tool, missing required args, bad type, enum, range, format, or `x-one-of`. |
| `utils.py` | Canonical state and `state_hash` (sha256). Drops `_meta`, info-only apps, and free-text fields before hashing. |
| `apps/<app>.py` | One module per app. `smart_home.get_status` is `SmartHome.get_status` in `apps/smart_home.py`. |
| `apps/base.py` | Shared read, create, patch, and delete helpers on one app’s state. |
| `apps/sys.py` | `sys.ask_missing_info`. Records the question; it does not change app data. |
| `apps/_*.py` | Shared helpers, not tools. |

Session rules that the agent code has to follow:

- One `ToolEnv(split, question_id, root)` per sample. `root` is the directory that contains the split folders (`train+public_test/`).
- Create that session once. A second session resets state and hides calls that still count.
- `env.validate(name, arguments)` checks schema and does not run the tool.
- `env.call(name, arguments)` runs it. A schema failure returns `{"ok": false, "error": ...}` and does not change state, but the call is still recorded.
- A missing id or an empty search is a normal tool result (`ok: true` with `done: false` or an empty list), not a schema error.
- `env.get_tool_calls()` and `env.get_state_hash()` are the two submission fields.
- Info apps (weather, gold price, fuel price, and similar) are readable and are left out of the compared state.

CLI, from inside a split directory:

```bash
python -m zaic_tools --split public_test --root .. --apps
python -m zaic_tools --split public_test --root .. --validate submission.jsonl
```

## Agent code

### `src/`

The speech-to-action pipeline. It imports `zaic_tools` from the split it is running.

- `asr.py` — transcribes `<id>.wav` and any history wavs the latest turn depends on. Writes the cache and reads it on the next run. On train, `audio_text` is the gold transcript used to separate speech errors from planning errors.
- `context.py` — turns `context` and `history` into the text block the planner sees: now, timezone, home, work, city, installed tool names, and prior turns. Date phrases such as “ngày mai” are resolved from `current_time` here when the value is explicit.
- `agent.py` — the step loop at temperature 0. Each step asks the local model for one next action, validates it, calls `env.call`, and appends the tool result. It stops on a finish action, on `sys.ask_missing_info`, or at the step cap. The model sees only tools installed on that sample.
- `validate.py` — our guard before `env.call`: tool is installed, arguments match the catalog, and a write that needs an id must have seen that id in context or in an earlier tool result. `env.validate` remains the organizer’s schema check.
- `run.py` — entry point. Runs one sample or every line of a split and writes JSONL:

```json
{"id": "public_0002", "tool_calls": [], "state_hash": "<from env.get_state_hash()>"}
```

`tool_calls` must be exactly `env.get_tool_calls()`, including rejected calls, because that is what gets replayed.

### `configs/`

`agent.yaml` holds the choices that must stay fixed between a local score and a submission: ASR checkpoint, planner checkpoint, temperature, max steps, split name, kit root, cache dir, and output path.

### `scripts/`

`score_dev.py` runs the agent on a held-out slice of train and compares the session with `label`. It reports state-hash match and call-level agreement, and tags each miss (wrong tool, extra call, missing lookup, bad argument, should have asked, unsupported). Holdout ids live next to this script so the same slice is reused every day.

### `cache/transcripts/`

One transcript file per wav, named by sample id. Safe to delete; `asr.py` rebuilds it. Keeps the planner loop from loading PhoWhisper on every prompt change.

### `outputs/`

Generated artifacts: `submission.jsonl` for a split, and per-run logs of tool traces used to tag errors. The public leaderboard consumes the JSONL. A later Docker package must write the same filename.

### `models/`

Local weights only (`PhoWhisper`, `Qwen2.5-7B-Instruct` or `Qwen3-8B`). The runtime loads paths from `configs/agent.yaml`. These files stay on disk and out of git.

## What each part is allowed to touch

| Path | Reads | Writes |
|---|---|---|
| `src/asr.py` | `audio/` | `cache/transcripts/` |
| `src/context.py` | `context`, `history`, catalog | nothing |
| `src/agent.py` | catalog, context text, tool results | one `ToolEnv` session in memory |
| `src/run.py` | `<split>.jsonl` | `outputs/submission.jsonl` |
| `scripts/score_dev.py` | train labels, agent output | `outputs/` score report |
| `zaic_tools` | `state.json`, catalog | in-memory state only |
