# ChronoMiner Test Suite

## Layout

```text
tests/
├── conftest.py            # hermetic harness and shared fixtures
├── recording_events.py    # run events recorded for assertions
├── test_hermetic.py       # tests of the harness itself
├── test_logging_setup.py  # logging is configured at run time, not on import
├── test_example_schemas.py, test_schema_invariants.py
│                          # the shipped example schemas
├── test_shipped_classifiers.py  # the shipped example classifiers
├── matrix_golden.py       # golden JSON of the request and image matrices
├── batch/                 # batch submit, check, finalize, reconcile, cancel
│                          # and repair on the store, over fake batch APIs
├── classify/              # the classify commands over a fake decision API,
│                          # reuse, refusals and the entry staleness triggers
├── cli/                   # the console script: help goldens, commands, warnings
├── common/                # chronominer/common/: hash check, SQLite, store,
│                          # capture, selection and the batch backends
├── core/                  # plan and run, spec, argv to spec, settings, events,
│                          # schema resolution, discovery, extension seam
├── extract/               # chunks, page sources, image settings, requests,
│                          # response parsing and the extraction units
├── images/                # chronominer/common/ image and native PDF helpers
├── llm/                   # model clients, options, capabilities, prompts,
│                          # retries and attempt capture
├── output/                # entries table, resume classification, the
│                          # _output.json, CSV and _line_ranges.txt renderers
├── ranges/                # line ranges: lines, matching, decision, generate,
│                          # adjust with a scripted model
├── ui/                    # the wizard: steps, sessions, flows, parity with the CLI
└── characterization/      # end-to-end runs compared with golden files
    ├── adapter.py         # run_tool(), the only entry into the tool
    ├── fakes*.py          # fake chat model and fake batch services
    ├── seams.py           # swaps an external object wherever the tool binds it
    ├── inputs.py          # generated ledgers, PDFs, image folders, mixed folders
    ├── golden.py          # snapshot normalization and golden comparison
    ├── stages.py          # multi-stage cases, each run twice
    ├── test_*.py          # one file per flow (see below)
    ├── context/, schemas/ # project context and fixture schemas of the cases
    └── golden/<group>/<case>/
```

Test files follow `test_<module>.py` naming; find the tests for a module by its
name. Fixture data sits beside its tests: `ranges/data/`, `llm/data/` and the
`golden/` folders of `cli/`, `ui/`, `llm/` and `extract/`.
`llm/test_request_matrix.py` and `extract/test_image_presets_matrix.py` compare
the requests and image settings of a matrix of providers, models and options with
their golden JSON (`CM_UPDATE_GOLDEN=1` rewrites it).

## Hermetic Harness

`tests/conftest.py` isolates every test from the machine it runs on. At
session start it removes every environment variable whose name contains
`API_KEY`, points `HOME`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`,
`HOMEDRIVE`, `HOMEPATH` and the `XDG_*` variables at a session temp directory,
and seeds a session tiktoken cache from the local one, so encodings are never
downloaded.

The autouse `_hermetic` fixture then gives each test its own copy of that
setup under `tmp_path`:

- the home variables, `TEMP`, `TMP`, `TMPDIR` and `tempfile.tempdir`;
- the default settings file (`config/settings.yaml`), so a real settings file
  in the repository is never read;
- the default logs directory.

The same fixture blocks connections to any non-loopback address, refuses to
start ChronoMiner (the `chronominer` program or package, run with `-m` or as a
script) or a clipboard program as a subprocess, and removes the log handlers a
test installed. A write guard records every file a test opens for writing; a
write outside the pytest base temp directory fails the test. The
`write_guard` fixture lets a test consume an expected write, as the golden
update does.

## Running Tests

Run from the repository root. Pass `--basetemp` when two runs share the
worktree, so their temp directories do not collide.

```powershell
# Full suite
uv run python -m pytest

# One directory, file or test
uv run python -m pytest tests/ranges/
uv run python -m pytest tests/output/test_csv.py -v
uv run python -m pytest tests/characterization/test_run_control.py -q `
    -p no:cacheprovider --basetemp "$env:TEMP\cm-run-control"

# Coverage, with the data file kept outside the repository
$env:COVERAGE_FILE = "$env:TEMP\chronominer.coverage"
uv run python -m pytest -p no:cacheprovider --basetemp "$env:TEMP\cm-cov" `
    --cov=chronominer --cov-report=term
```

Test dependencies are part of the dev extra: `uv sync --all-extras`. No test
calls a real provider; the harness removes every API key from the environment
and blocks non-loopback connections.

## Characterization Tests

The tests in `tests/characterization/` run the real CLI code paths in process
and compare everything a run produces with golden files.

- `run_tool(command, *, llm, root, config=None, env=None, batch_service=None,
  **keywords)` in `adapter.py` is the only function that knows how the tool is
  invoked. It translates the case vocabulary (`extract`,
  `generate_line_ranges`, `adjust_line_ranges`, `check_batches`,
  `cancel_batches`, `repair` and keywords such as `schema`, `input` and
  `chunking`) into `chronominer` arguments; `--json` and `--settings` are
  always passed. The returned `RunResult` holds the exit code, the JSON
  summary line, the console output, the files the run created and the
  requests the fake chat model received.
- The autouse `fake_llm` fixture replaces the provider classes with a
  scripted fake chat model, so the real provider, retry and parsing code runs
  without a network. Text requests are matched to scripts by the record
  headings of the generated ledgers, page images by a gray-level marker drawn
  into each page. `fake_batch` installs fake OpenAI, Anthropic and Google
  batch services behind the real batch backends.
- Each case runs twice in fresh roots, and both snapshots must match before
  the second is compared with `golden/<group>/<case>/`. A snapshot holds
  every created file (`_output.json`, CSV and `_line_ranges.txt` as text,
  `chronominer.sqlite` as a dump of its tables and the `entries`, `chunks`
  and `ranges` views; paths, datetimes, durations, versions and hashes
  normalized to placeholders), `summary.json` (exit code and JSON line) and
  `requests.json` (client constructions and request digests).
- To accept a deliberate change, rerun the affected test file with
  `CM_UPDATE_GOLDEN=1`, which rewrites its golden cases, and review the diff.
  `CM_GOLDEN_RENDERED_ONLY=1` compares only the rendered files.

| File | Flow |
| ---- | ---- |
| `test_smoke.py` | one case per command |
| `test_sources.py` | text files and folders, images, PDFs, mixed folders, layouts |
| `test_chunking.py` | `--chunking auto`, `line-ranges` and `adjust` |
| `test_line_ranges.py` | `ranges generate` |
| `test_adjust.py` | `ranges adjust` with scripted boundary answers |
| `test_providers.py` | provider classes, model options and image messages |
| `test_response_formats.py` | enforced and prompted JSON |
| `test_context.py` | text context resolution and context images |
| `test_outputs.py` | `_output.json` and CSV shapes, a failed chunk |
| `test_batch.py` | batch submit and finalize through the fake batch services |
| `test_cancel.py` | `batch cancel` |
| `test_repair.py` | `repair`, synchronous and in batches |
| `test_run_control.py` | resume, force, dry run, JSON line, exit codes |
| `test_golden_mode.py` | the normalizer and the golden update modes |

## Markers

Registered in `pyproject.toml`; `--strict-markers` is enforced and no marker
is deselected by default.

| Marker | Description |
| ------ | ----------- |
| `unit` | Fast unit tests without external dependencies |
| `integration` | Tests across modules, with real files and fakes |

Async tests need no marker: `asyncio_mode = "auto"` is set.

## Fixtures

Shared fixtures and builders are in `tests/conftest.py`: `write_guard` from
the harness, the `make_spec`, `make_settings` and `make_job` builders, temp
directories and sample text, PDF and image inputs, and the environment
fixtures `mock_env_no_api_keys` and `mock_env_with_openai_key`. The
characterization fixtures (`fake_llm`, `fake_batch`, `golden`, `case_root`,
`project_context`) are in `tests/characterization/conftest.py`. Consult both
files for the authoritative list.

## Writing New Tests

- Name files `test_<module>.py`, functions `test_*`; markers are strict.
- Write only under `tmp_path`; the write guard fails any other write.
- Mock external services; no test calls a real LLM endpoint.
- Run the tool in characterization tests only through `run_tool`.
