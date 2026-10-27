# ChronoMiner v1.0.4

ChronoMiner extracts structured data from text files, images and PDFs with language
models. A JSON schema describes the entries to find, such as the items of a price list
or the households of a census return; the model reads each chunk of a source and
returns the entries as JSON. Model calls run on OpenAI, Anthropic, Google and
OpenRouter models and on OpenAI-compatible endpoints that you name in the settings.
The commands serve scripts and agents; `chronominer ui` walks you through a run in the
terminal. Please report bugs on the
[GitHub issue tracker](https://github.com/Paullllllllllllllllll/ChronoMiner/issues).

## Overview

An item is one text (`.txt`, `.md`), image or PDF file, cut into chunks: the line
ranges of a text, the image, the pages of a PDF. Each chunk is one model request and
one unit of the item's document in the database `chronominer.sqlite` of the output
folder, committed with the provider response and its parsed entries as soon as the
answer arrives. `{stem}_output.json` and, on request, a CSV are rendered from the
database, so an interrupted run resumes from it and `render` rewrites the files
without model calls. For `prices.txt` and `--output out`, a run writes
`out/prices_output.json` and `out/chronominer.sqlite`.

## Installation

ChronoMiner needs Python 3.13 or later and [uv](https://docs.astral.sh/uv/). Install it
from a clone, since the settings file and the schemas are read from the repository.

```bash
git clone https://github.com/Paullllllllllllllllll/ChronoMiner.git
cd ChronoMiner
uv sync --all-extras          # or uv sync, plus --extra dev for the tests
mkdir config
cp chronominer/resources/settings.example.yaml config/settings.yaml
export OPENAI_API_KEY="sk-..."   # PowerShell: $env:OPENAI_API_KEY = "sk-..."
```

The `dev` extra holds pytest, ruff and mypy. The settings name the API key variable of
each provider (`api_keys:`); the defaults are `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY` and `OPENROUTER_API_KEY`.

## Quick Start

```bash
uv run chronominer extract --schema price_list_entries --input list.txt --output out
uv run chronominer extract --schema wage_records --input sources --all --formats csv
uv run chronominer ranges generate --input texts --chunk-tokens 3000
uv run chronominer extract --schema trade_records --input portbook.txt --chunking adjust
uv run chronominer extract --schema census_households --input scans --select 1-3 --dry-run --json
uv run chronominer ui
```

Bare `chronominer` prints the commands, `<command> --help` the options of one. A
folder that holds several items needs `--all`, `--select` or `--files`. Relative
paths resolve against the current directory; with `uv --directory <clone> run ...`
that is the clone, so pass absolute paths.

## Command Reference

The tables list every option of each command with its `--help` wording and default.
They mirror the option tables in `chronominer/spec.py`, and
`tests/cli/test_readme_options.py` checks them against those tables. Options without a
stated default are off or unset until given. The last column says whether the settings
`defaults:` block can set the option; the key is the flag without dashes, with
underscores for hyphens (`chunk_tokens`). A list option given twice keeps the values of
both.

### extract

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | text (.txt, .md), image or PDF file, or a folder of these | | no |
| `--type {auto,text,images,pdf}` | input type; auto takes the text, image and PDF files of a folder and detects a file's type by its extension | auto | yes |
| `--select EXPR` | select items by number (1,3), range (1-5) or name search | | yes |
| `--all` | process every discovered item | | yes |
| `--files LIST` | comma-separated names of files inside the input folder | | no |
| `--recursive`, `--no-recursive` | search the subfolders of the input folder | off | yes |
| `--pages RANGE` | chunks (PDF pages, line ranges): 5, first:5, last:5, 3-7, 3-, -7 or 1,3,5-8 | all | no |
| `--output DIR` | output folder | beside the input | yes |
| `--beside-input` | write outputs beside each input | | no |
| `--layout {flat,mirror}` | output folder layout: flat (every output in the output folder) or mirror (the input folder tree) | flat | yes |
| `--formats LIST` | comma-separated formats rendered besides {stem}_output.json and chronominer.sqlite; from csv | none | yes |
| `--schema NAME\|PATH` | extraction schema: a file path, a file name in schemas/ without .json, or the name field of a schema there | required | yes |
| `--context TEXT\|PATH\|auto\|none` | extraction context: topics as text or a path to a text file (overrides sidecars), auto (the input's {stem}_extract_context.txt, else the folder's context file) or none | auto | yes |
| `--fallback-context TEXT\|PATH\|none` | extraction context for inputs without a sidecar under --context auto: topics as text, a path to a text file, or none | none | yes |
| `--context-image PATH\|auto\|none` | image sent with every chunk: an image file, auto (the input's {stem}_extract_context image) or none | none | yes |
| `--chunking {auto,line-ranges,adjust}` | text chunking: auto (by --chunk-tokens), line-ranges (the input's {stem}_line_ranges.txt) or adjust (line ranges whose boundaries the model moves first); adjust rewrites that file beside the input, not under --output | auto | yes |
| `--chunk-tokens N` | target tokens per text chunk, also for line ranges written anew | 3500 | yes |
| `--context-window N` | lines around each range boundary sent to the model; only for chunking adjust | 6 | yes |
| `--provider {openai,anthropic,google,openrouter,custom}` | model provider | inferred from the model name | yes |
| `--model NAME` | model name; a provider: prefix (anthropic:NAME) selects the provider | gpt-6-luna | yes |
| `--endpoint NAME` | named custom endpoint from the settings (implies provider custom) | none | yes |
| `--reasoning-effort {none,minimal,low,medium,high,xhigh,max}` | reasoning effort | xhigh | yes |
| `--verbosity {low,medium,high}` | output verbosity | medium | yes |
| `--max-output-tokens N` | maximum output tokens | the model's output limit | yes |
| `--temperature T` | sampling temperature, 0.0 to 2.0 | 1.0 | yes |
| `--top-p P` | nucleus sampling, 0.0 to 1.0; sent only when supported | 1.0 | yes |
| `--service-tier {auto,default,flex,priority}` | OpenAI service tier; only for provider openai | default | yes |
| `--response-format {schema,json}` | schema (API-enforced) or json (prompted, validated) | schema when the model supports it, else json | yes |
| `--dpi native\|N` | render resolution: native or a DPI value | the provider's recommended value | yes |
| `--image-detail {low,high,auto,original}` | image detail sent to the model | the provider's recommended value | yes |
| `--image-format {jpeg,png}` | page image format | the provider's recommended value | yes |
| `--jpeg-quality N` | JPEG quality, 1 to 100 | the provider's recommended value | yes |
| `--grayscale`, `--no-grayscale` | convert page images to grayscale | the provider's recommended value | yes |
| `--batch`, `--no-batch` | submit the chunks to the provider's batch API | off | yes |
| `--force` | extract inputs again from the start: every stored unit of the input is reset, also outside --pages; stored responses are kept | | no |
| `--concurrency N` | parallel requests | 16 | yes |
| `--dry-run` | plan without API calls or writes | | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

`--schema` and `--input` are required, and `--beside-input` overrides an `output`
settings default. A folder holding several items needs one of `--all`, `--select`
and `--files`, which exclude each other; without one the run exits 2 before any
call. `--select` numbers the items in the order the run discovers them (1-based) and
takes numbers and ranges (`1,3,5-8`) or, for any other text, keeps the items whose
path below the input folder holds it, ignoring case. An option whose condition does
not hold (`--context-window` with `--chunking auto`) has no effect, and the run warns
on stderr. The chunking options apply to text items, the image options to image and
PDF items. A dry run plans without the API key variables and names an unset one in a
warning.

### ranges generate

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | text file, or a folder of text files, to write line ranges for | | no |
| `--recursive`, `--no-recursive` | search the subfolders of the input folder | off | yes |
| `--chunk-tokens N` | target tokens per text chunk, also for line ranges written anew | 3500 | yes |
| `--model NAME` | model whose tokenizer counts the chunk tokens | gpt-6-luna | yes |
| `--dry-run` | plan without API calls or writes | | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

### ranges adjust

`ranges adjust` also takes the model options of `extract`, with their defaults and
settings defaults.

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | text file, or a folder of text files, whose line ranges to adjust | | no |
| `--recursive`, `--no-recursive` | search the subfolders of the input folder | off | yes |
| `--schema NAME\|PATH` | schema of the entries whose boundaries the model looks for: a file path, a file name in schemas/ or a schema's name field | required | yes |
| `--context-window N` | lines around each range boundary sent to the model | 6 | yes |
| `--context TEXT\|PATH\|auto\|none` | adjust context: topics as text or a path to a text file (overrides sidecars), auto (the input's {stem}_adjust_context.txt, else the folder's context file) or none | auto | no |
| `--fallback-context TEXT\|PATH\|none` | adjust context for inputs without a sidecar under --context auto: topics as text, a path to a text file, or none | none | no |
| `--pages RANGE` | line ranges: 5, first:5, last:5, 3-7, 3-, -7 or 1,3,5-8 | all | no |
| `--force` | adjust the line ranges again: the stored decisions are reset | | no |
| `--concurrency N` | parallel requests | 16 | yes |
| `--dry-run` | plan without API calls or writes | | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

### batch check

Without `--wait`, `batch check` checks every submission once; with `--wait` it checks
again as described under [Batch Processing](#batch-processing). `--adopt` and
`--abandon` act in the first check only, and the guided UI checks once. A finalized
document is rendered in the formats of the run that submitted it, plus those it was
rendered in before; `--formats` names other formats instead of the run's.

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input DIR` | folder searched for batch submissions, subfolders included | the current directory | no |
| `--schema NAME` | check only documents extracted with this schema | every schema | no |
| `--batch-ids IDS` | comma-separated batch IDs to check | every batch | no |
| `--adopt BATCH_ID` | record this provider batch for the unresolved submission it belongs to, then finalize it | | no |
| `--abandon SUBMISSION_ID` | give up an unresolved or submitted submission: its chunks return to pending and its uploaded files are deleted; a submitted batch is not cancelled | | no |
| `--wait` | check again every --poll-minutes until no batch work is pending, or until --wait-hours have passed | | no |
| `--poll-minutes N` | minutes between two checks with --wait | 15 | no |
| `--wait-hours N` | hours after which --wait starts no further check; a running check finishes | 50 | no |
| `--formats LIST` | comma-separated formats rendered besides {stem}_output.json and chronominer.sqlite; from csv | the formats of the run that submitted each document | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

### batch cancel

One of `--batch-ids`, `--all` and `--input` is required.

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--batch-ids IDS` | comma-separated batch IDs to cancel | | no |
| `--all` | cancel every unfinished batch of every provider | | no |
| `--input DIR` | cancel the open batch submissions recorded in the databases below this folder | | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

### repair

`repair` also takes the model and image options of `extract`, with their defaults and
settings defaults.

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | folder, chronominer.sqlite or input whose failed chunks to run again; subfolders included | the current directory | no |
| `--schema NAME` | repair only documents extracted with this schema | every schema | no |
| `--indices LIST` | 0-based chunk indices (order in the document) among the failed chunks: 0,5,12 or 3-7 | every failed chunk | no |
| `--batch`, `--no-batch` | submit the repairs to the provider's batch API | off | yes |
| `--dry-run` | plan without API calls or writes | | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

### render

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | folder or chronominer.sqlite to render | . | no |
| `--formats LIST` | comma-separated formats to render besides {stem}_output.json; from csv | the formats each document was rendered in | no |
| `--force` | replace files changed outside the tool | | no |
| `--dry-run` | plan without API calls or writes | | no |
| `--json` | print one JSON summary line on stdout | | no |

### drop

`--input` and exactly one of `--select`, `--files` and `--all` are required.
`--select` numbers the stored documents in name order; a `--files` entry names a
document or the file name of its input, and an entry that names two documents is
refused.

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | output folder holding chronominer.sqlite, or the database | | no |
| `--select EXPR` | select documents by number (1,3), range (1-5) or name search | | no |
| `--all` | drop every stored document | | no |
| `--files LIST` | comma-separated document names or input file names | | no |
| `--dry-run` | plan without API calls or writes | | no |
| `--json` | print one JSON summary line on stdout | | no |

### ui

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

## Exit Codes and JSON Summary

| Code | Meaning |
| --- | --- |
| 0 | every item completed or, under `--batch`, was submitted, or nothing was left to do |
| 1 | an item failed (refused items included), another run leases an item (the final message names each blocked item with its reason and a lease with its expiry; an item waiting for a batch is named but does not change the exit code), a repaired chunk stayed unrepaired, a batch failed or waits unresolved, `batch check --wait` reached `--wait-hours` with batches pending, a cancellation failed, a file could not be rendered or its document was held by another run, `drop` refused a document, or an unexpected error |
| 2 | usage or configuration error: invalid flags or settings, an unknown schema, an unset API key variable (outside a dry run), a provider without a batch API under `--batch`, a model without image input for image or PDF items, no items in the input, a folder of several items without `--all`, `--select` or `--files`, a `--select` or `--files` that matches no item or stored document, a missing input path or folder, no store at the `drop` input, two inputs that share one output, a database without schema version, newer than the tool or unreadable |
| 130 | interrupted (Ctrl+C, or Ctrl+Break on Windows) |

Each command's `--help` lists its own exit codes. Progress, warnings and errors go to
stderr, in color only on a terminal and when `NO_COLOR` is unset or empty. With
`--json`, stdout carries exactly one JSON line on every exit, errors and interrupts
included; `ui` writes none. Every line starts with `dry_run`, true whenever
`--dry-run` was given. Every field listed for a command is present on every exit path,
zero where nothing ran, unless stated otherwise.

| Command | Fields |
| --- | --- |
| `extract`, `ranges generate`, `ranges adjust` | `items_total`, `items_complete`, `items_failed`, `items_skipped`, `items_blocked`, `items_submitted`, `outputs`, `usage` |
| `batch check` | `command` (`check_batches`), `finalized`, `pending`, `failed`, `unresolved`, `submitted`, `complete`, `outputs`, `usage`; the booleans `adopted` with `--adopt` and `abandoned` with `--abandon`; with `--wait`, `polls` (checks made), `waited_seconds` and `stopped` (`settled`, `timeout`, `interrupted`, `error`, or null before the first check) |
| `batch cancel` | `command` (`cancel_batches`), `cancelled`, `failed` |
| `repair` | `command` (`repair`), `repaired`, `failed`, `usage` |
| `render` | `command` (`render`), `databases`, `documents`, `written`, `unchanged`, `kept`, `stale`, `skipped`, `outputs` |
| `drop` | `command` (`drop`), `dropped`, `refused`, `deleted_files`, `kept_files` |

`items_total` counts the items processed (on a plan error, those the run would have
processed), `items_skipped` the complete, unstored and blocked items and those of which
`extract --pages` selects no chunk (each named in a warning), `items_blocked` the
blocked items among them (those another run leases exit 1; those waiting for a batch
do not; `ranges` commands count a held file as failed, so it stays 0 there),
`items_submitted` the items whose chunks wait in batches for `batch check`
(`--batch`; they count neither complete nor failed), and `outputs` the absolute
paths written, the databases last. An item whose selected chunks all succeeded
while chunks outside `--pages` remain pending or failed counts neither complete nor
failed; stderr reports it as left partial by `--pages`. `usage` holds the tokens of
each role that made calls (`extract`, and `adjust` for boundary decisions) and their
sum under `total`, each as `input_tokens`, `output_tokens`, `total_tokens`,
`cached_tokens` and `reasoning_tokens`, also after an interrupt. A successful dry run
adds `to_process` (each item's `name`, `type` and resume `state`), `skipped`,
`unstored`, `blocked` and `refused` (`name` and `reason`), and `spec`, which maps
every field of the effective run specification, such as `model.model`, to its
`value` and `source` (`flag`, `settings`, `default` or `resolved`, described under
[Settings File](#settings-file)); the ranges commands add `to_process` and `spec`
only.

For `batch check`, the first four count submissions by outcome (`unresolved`: waiting
for `--adopt` or `--abandon`), `submitted` the submissions a check opened (always 0,
since a check only collects), `complete` the finalized documents left with no pending
or failed chunk, `outputs` the files rendered and `usage` the tokens of the finalized
results under `extract`; a check that finalizes a failed chunk exits 1. For
`render`, `kept` counts files changed outside the tool and left alone, `stale` files
that could not be replaced and `skipped` documents another run holds. For `drop`,
`dropped` names the documents deleted (in a dry run, those it would delete),
`refused` lists each document left in place as `doc_id`, `name` and `reason`,
`deleted_files` the files removed and `kept_files` each file left in place as `path`
and `reason`.

When the environment variable `CHRONO_HEARTBEAT_FILE` names a file, `extract` and
`batch check --wait` also write their progress there as one JSON object for a job
launcher: `units_done` of `units_total` `unit_label`, `eta_seconds` and `updated_at`
(ISO 8601, UTC). An extraction counts the current item's chunks and names the item's
position (`chunks processed, item 3/12`); its ETA, from the chunk rate, is given only
during the last item and is null otherwise. A wait counts the submissions finished of
those it has seen (`batches`) before each sleep. The file is replaced at most every 15
seconds, and a write that fails is skipped without affecting the run.

## Guided UI

`chronominer ui` asks first for the task, one of the commands above. An extraction
then asks for the input, the type and, for a folder of several items, `Which items?`
(all, by number or range, by name search, or from the list), then chunking, the units
to extract (`Which pages?`, `Which line ranges?` or `Which chunks?`: all, or a range in
the `--pages` syntax), output and formats, model, prompt and context (response
format, schema, context and context image), page images and the run options, whose
`Parallel requests:` is asked only for a synchronous run; a step appears only for the
inputs that use it. The context is `Automatic (sidecar)`, with a fallback for items
without a sidecar, `Topics as text`, `A text file` or `None`. Every list offers Back,
a text or path prompt takes `<` as Back, settings defaults are preselected, and
answers act like flags over them; an input that holds nothing to process asks for the
path again. The review screen marks settings defaults with `(s)`, names on its
`access` line how the model calls authenticate (the provider's API key variable),
shows the condition warnings, previews the plan without API calls and prints the
equivalent command. It offers Run, Edit a step, Copy command and quit, and Quit; Run
is withheld while the plan fails and while an API key variable the run needs is unset
(the review names it), though a dry run works without keys. The UI needs an
interactive terminal, else it exits with code 2; otherwise its exit codes are those
of the chosen task's command.

## Settings File

`--settings FILE` reads the named file, which must exist. Otherwise ChronoMiner reads
`config/settings.yaml` in the repository (gitignored), else the packaged
`chronominer/resources/settings.example.yaml`, and then logs one line asking you to
copy it. Every key is optional; unknown keys are an error that names the file and the
key. The example file:

```yaml
# ChronoMiner settings: machine values and optional run defaults.
#
# Copy this file to config/settings.yaml in the repository (gitignored) and
# edit the copy, or pass another file with --settings FILE. Every key is
# optional; a missing key keeps the value shown here. Unknown keys are errors.
# Relative paths resolve against the current directory.

# Environment-variable NAMES that hold each provider's API key (never the key).
api_keys:
  openai: OPENAI_API_KEY
  anthropic: ANTHROPIC_API_KEY
  google: GOOGLE_API_KEY
  openrouter: OPENROUTER_API_KEY

# Named OpenAI-compatible endpoints, selected with --endpoint NAME.
endpoints: {}
#  example:
#    base_url: https://your-endpoint.example.com/v1/
#    api_key_env: YOUR_ENDPOINT_API_KEY
#    supports_vision: true   # the model accepts images
#    supports_schema: false  # the API enforces JSON schemas

# HTTP timeouts in seconds. request is the read timeout; 900 s suits the
# OpenAI flex tier, which may queue requests.
timeouts:
  request: 900
  connect: 10
  write: 30
  pool: 30

# Client-side rate limits per provider as [max_requests, window_seconds].
rate_limits:
  - [120, 1]
  - [15000, 60]
  - [15000, 3600]

# Retries of failed API calls (429, 5xx, timeouts, connection errors).
retry:
  max_attempts: 8         # every call, the first included
  timeout_attempts: 3     # calls that end in a timeout (billed, no usage)
  validation_attempts: 1  # calls that return unparseable structured output
  min_input_tokens: 0     # retry image requests reporting fewer input tokens
  # Wall-clock limit of one chunk across all attempts: auto (request timeout
  # times timeout_attempts plus 300 s), off, or seconds.
  page_timeout: auto

# Pacing of the --concurrency workers (the LLM chunk requests).
concurrency:
  delay_between_tasks: 0  # seconds each worker pauses before its next request

# Long-side limit in pixels of OpenAI pages sent at original detail; null
# keeps the model's default (3000 for gpt-6-luna, none for other models).
images:
  original_max_side_px: null

# Provider prompt caching of the static prompt (instructions, schema, context).
prompt_caching:
  enabled: true
  anthropic:
    ttl: 5m  # 5m or 1h
  openai:
    prompt_cache_retention: null  # or 24h: documented models; GPT-5.6+ get ttl 30m
  openrouter:
    anthropic_cache_control: true

# Folder of the state files of extensions; '' means ~/.chronominer.
state_dir: ''

# Log folder, or a file ending in .log; '' writes logs/application.log in the
# repository.
logs_dir: ''

# Run defaults keyed by option name (flags override them), for example:
# defaults:
#   model: gpt-6-luna
#   reasoning_effort: xhigh
#   service_tier: default
#   formats: [csv]  # chronominer.sqlite and {stem}_output.json are always written
#   layout: flat
#   chunk_tokens: 3500
#   concurrency: 16
#   batch: false
#   fallback_context: none  # inputs without a sidecar: topics as text or a file
```

A failed call waits 2 s, doubled per retry, plus up to 1 s of jitter, at most 120 s, a
server's `Retry-After` included. The `defaults:` block takes the option names of the
`extract` table; the other commands use the defaults of the options they share with
it. Options that only the command line sets and invalid values are errors there.
Precedence is flag (or wizard answer), then settings default, then code default;
`--dry-run --json` prints the source of each value, and the database records the
effective spec of every run. In the dry run's spec an unset output limit, response
format and (for image inputs) image option shows the value the requests use, with
source `resolved` (the preset of new documents; a resumed document keeps the image
settings it was started with), and a provider other than OpenAI shows no service
tier. Dry runs write no log file.

## Providers and Response Formats

The provider comes from `--provider`, else a `provider:` prefix on the model name
(`anthropic:claude-sonnet-5-5`), else the provider rules of the shared capability
table, so that a model runs on the provider its capabilities are resolved for:
Google's `models/` prefix, `gemini` and `gemma` are Google, a `vendor/model` name is
OpenRouter, `claude` is Anthropic, the GPT, o-series and `whisper` names are OpenAI,
and bare `llama`, `mistral`, `mixtral`, `qwen` and `deepseek` names are OpenRouter;
any other name runs on OpenAI. A provider set at a higher precedence level than a
model of another provider is an error. `--endpoint NAME` selects an endpoint from
`endpoints:` and implies provider `custom`; the endpoint declares whether its model
takes images (`supports_vision`) and whether its API enforces JSON schemas
(`supports_schema`).

`--response-format schema` sends the selected schema with the request, and `json` puts
it into the prompt and validates the answer's required keys with retries; both work on
every provider. Unset, a model with structured output (or an endpoint with
`supports_schema`) uses `schema`, any other `json`; a requested `schema` on a model
without structured output falls back to `json` and logs a warning, while a custom
endpoint keeps the requested format. In both formats an answer that holds no JSON
object (a bare number, plain text) fails its chunk, in synchronous and batch runs.
OpenAI receives the schema as a strict `text.format`, custom endpoints as a strict
`response_format`, Anthropic as a `json_schema` structured output with union types
reduced to one type, and Google as `response_schema`. OpenRouter receives a
`response_format` without `strict`, so enforcement depends on OpenRouter and the
routed model.

The model capability table in `chronominer/common/capability_table.py`, shared with
AutoExcerpter and ChronoTranscriber, feeds the resolver in
`chronominer/common/capabilities.py`. Its family rules follow the vendors'
documentation (read 13.10.2026): each record of `common/capability_facts.py` adds,
drops or re-matches a rule, or sets one of its fields, and names its source. The
first rule matching the
lowercased model name sets image input, structured output, reasoning and the
documented effort levels, verbosity, sampling, the Flex tier and output limits. An
unmatched name, or one the vendor does not document (`gpt-5.3`, `gpt-5.3-chat`,
`glm-flash-latest`), gets a conservative profile without image input or structured
output, so a run on images or PDFs stops at planning.

Synchronous calls and batch bodies send the same OpenAI options. The service tier
reaches OpenAI only (synchronous calls send it to reasoning models only), and `flex`
only the models OpenAI lists for Flex processing (GPT-6, GPT-5.6, 5.5, 5.4, 5.2, 5.1
and 5 with the listed mini, nano and pro variants, o3, o4-mini); other models get no
tier, with one warning per run. An effort a model does not document becomes the next
level it does, or its highest, with one warning per run: for example `none` and
`minimal` become `low` on gpt-6-astra and gpt-6.1-sol, and Gemini `minimal` becomes
`low` on Gemini 3.1 Pro and 3.7/3.8 Flash. The verbosity reaches the GPT-5 and GPT-6
reasoning models only; the chat-latest models are not reasoning models. A set
`prompt_cache_retention` goes to the models documented for extended retention, and
GPT-5.6 and later get `prompt_cache_options` with a `30m` TTL instead, in synchronous
and batch requests alike; other models get neither. Unset image options take the
preset of the model's family (OpenRouter models that of the family their name
names, Claude, Gemini or OpenAI; other OpenRouter models the custom preset); every
family sends colour JPEG at quality 95:

| Family | `--dpi` | `--image-detail` | Format | JPEG quality | Grayscale |
| --- | --- | --- | --- | --- | --- |
| openai | native | original | jpeg | 95 | no |
| anthropic | native | auto | jpeg | 95 | no |
| google | 300 | high | jpeg | 95 | no |
| custom | 150 | high | jpeg | 95 | no |

An item whose pages are stored keeps the stored DPI, format, JPEG quality and
grayscale for every image option the run leaves unset, so it resumes under the
values it was started with. `native` renders each PDF page at the density of its
scan. At `original` detail, gpt-6-luna pages are limited to 3,000 px on the long
side; `images.original_max_side_px` in the settings sets this limit for every OpenAI
model.

## Schemas

`--schema` resolves, in this order: an existing file path; a file `<name>.json` in the
`schemas/` folder of the repository; a schema in that folder whose `name` field equals
the argument (`--schema PriceListEntries`). An unknown name exits 2 and lists the
available schemas. The repository ships eight example schemas:

| Schema | Sources |
| --- | --- |
| `price_list_entries` | price lists, price currents and market tariffs: item, quantity, unit, price, currency, date, place |
| `wage_records` | payrolls and wage books: person, occupation, wage, period, currency, date, employer |
| `trade_records` | port books, customs records and merchant accounts: vessel or merchant, port, cargo, quantity, value, currency, date |
| `probate_inventories` | probate and estate inventories: decedent, occupation, date, place, currency, items with appraised values |
| `census_households` | census returns and population registers: household, person, age, sex, occupation, relation to head, address |
| `parish_register_events` | baptisms, marriages and burials: event type, date, parish, persons and roles, officiant |
| `bibliographic_catalogue_entries` | catalogues, bibliographies and book lists: author, title, place, publisher, year, format |
| `correspondence_metadata` | letters and letter calendars: sender, recipient, date, place, language, summary |

Every example entry also has `source_page` and `notes`. To write your own schema, copy
one of these files: it holds `name`, `description` and, under `schema`, an object with
a boolean `contains_no_content_of_requested_type` and an `entries` array. Each
top-level property of the entry object becomes a column of the `entries` view and of
the CSV. Save the file in `schemas/` to call it by name, or pass its path.

## Inputs and Chunking

With `--type auto`, a folder yields its text, image and PDF files in natural order
(with `--recursive` those of its subfolders too, except top-level `output` and
`*_outputs` folders), and a file's type comes from its extension; a file of no known
type is text. `--type text`, `images` or `pdf` keeps the files of that type. Files the
tool reads or writes beside an input (context files, line ranges, outputs and the
database) are never items. An image is one chunk and a PDF has one chunk per page. A
text is cut into line ranges by `--chunking`:

| `--chunking` | Chunks of a text |
| --- | --- |
| `auto` | ranges of about `--chunk-tokens` tokens, cut in memory |
| `line-ranges` | the ranges of `{stem}_line_ranges.txt` beside the input, written from `--chunk-tokens` when missing |
| `adjust` | the same file after the model has moved its boundaries to entry boundaries, as `ranges adjust` does; the file is rewritten in place beside the input, also when `--output` names another folder |

A chunk grows line by line until the next line would pass `--chunk-tokens` (counted
with the model's tiktoken encoding, else `cl100k_base`). `--pages` selects chunks by
their 1-based position: PDF pages, or the line ranges of a text; such a run leaves the
document partial. `--context` takes topics as text or a path to a text file: a path
to an existing file is read, anything else is the context. With `--context auto`, an
item's context is `{stem}_extract_context.txt` beside the file, else
`{folder}_extract_context.txt` beside its folder, else `--fallback-context` (or
`fallback_context` in `defaults:`); without one the item has no context.
`--context-image auto` looks for `{stem}_extract_context.{png,jpg,...}` the same way,
else for `context/extract_context.{png,jpg,...}` in the repository. The context image
is sent before every chunk to OpenAI and OpenRouter models; for other providers it is
skipped with a warning in the log.

## Line Ranges

`ranges generate` writes `{stem}_line_ranges.txt` beside every text file of its input.
Each line holds one range as `(start, end)`, 1-based and inclusive. The file is an
input: you can edit it by hand, and `--chunking line-ranges` extracts one chunk per
range. `ranges adjust` asks the model, for each range, where the entries of the schema
begin in a window of `--context-window` lines around the range start, and moves,
keeps or deletes the boundary. Each decision is stored at once, and the file is then
rendered again from the store. A file the store rendered last, or whose ranges equal
the stored starting ranges, resumes the stored decisions; any other file (written anew
or edited) is a new starting point, as `--force` and changed adjust settings are. A
range whose model calls all failed is decided again by the next run. Both commands
record the ranges in the `chronominer.sqlite` beside the text file; `extract
--chunking adjust` runs the same adjustment in the database of the extraction. The
adjust context follows the lookup of [Inputs and Chunking](#inputs-and-chunking) with
the suffix `_adjust_context.txt`; under `extract --chunking adjust` a `--context` text
or file and the `--fallback-context` serve the boundary decisions too.

## Output Contract

For a folder input `in/` holding `prices.txt` and `sub/book.pdf`, `--output out`
writes:

```text
--layout flat (default)   out/prices_output.json, out/book_output.json
--layout mirror           out/prices_output.json, out/sub/book_output.json
--beside-input            in/prices_output.json, in/sub/book_output.json
```

The database `chronominer.sqlite` lies in the output root: the output folder for both
layouts (`out/chronominer.sqlite`), or each input's own folder with `--beside-input`
(`in/chronominer.sqlite` and `in/sub/chronominer.sqlite`). A single-file input
`in/sub/book.pdf` writes `out/book_output.json` under either layout, and
`in/sub/book_output.json` beside the input, the default without `--output`. Two inputs
whose outputs would share one path (`a/list.txt` and `b/list.txt` under `--layout
flat`) stop the run at planning with exit code 2; use `--layout mirror` instead.

`{stem}_output.json` is always written once a chunk is done or failed, as
`{"_chronominer_metadata": {...}, "records": [...]}`. Each done chunk is one record,
in chunk order: `custom_id`, `chunk_index` (1-based), `chunk_range` (`[start, end]` of
a text chunk, else null), `response` (its `output_text` is the answer as received)
and, for an image or PDF page, `image_provenance`. `_chronominer_metadata` holds
`schema_name`, `model_name`, `chunking_method`, `total_chunks`, `timestamp`, `version`
and, when known, `partial`, `failed_chunks`, `batch_tracking` and further run
metadata. `render` keeps a file into which another tool wrote metadata.

`--formats csv` adds `{stem}_output.csv`: one row per entry, columns in the order of
the schema's entry object, an object field as one column per property named with dots
(`place.town`), lists of scalars joined by `"; "` and other nested values as JSON. The
file is UTF-8 with a byte-order mark and holds the header alone when the document has
no entries. `{stem}_line_ranges.txt` is the third rendered view, re-rendered from the
range units of a text. The context sidecars `{stem}_extract_context.{txt,png}` and
`{stem}_adjust_context.txt` are inputs and are never written.

## Store

Every output root holds one `chronominer.sqlite`, written by every run: the runs with
tool version and effective spec, the prompt and schema texts, the documents and their
units, every provider attempt verbatim, the parsed entries, the batch submissions and
the registry of rendered files. Readers outside the tool use the documented views,
which stay stable; the tables behind them may change.

| View | One row per | Columns |
| --- | --- | --- |
| `entries` | extracted entry of a done chunk, by document, chunk and entry | `document`, `chunk_index`, `entry_index`, `schema`, `model`, `provider`, `run_id` |
| `chunks` | chunk unit, by document and chunk | `document`, `chunk_index`, `source_index`, `label`, `status`, `model`, `provider`, `usage`, `output_text`, `error`, `error_type`, `custom_id`, `stale`, `run_id` |
| `ranges` | line range of a text, by document and range | `document`, `range_index`, `original_start`, `original_end`, `adjusted_start`, `adjusted_end`, `should_delete`, `status` |

A document is named by its `{stem}_output.json` path relative to the output root,
without that suffix (`sub/book`). In `entries`, one column per top-level field of the
schema's entry object sits between `schema` and `model`, typed TEXT, INTEGER or REAL
(booleans as 0 or 1, objects and lists as JSON text); a field named like a fixed
column appears as `field_<name>`. Field columns are added when a schema first writes
entries, and the fields of other schemas are NULL. In `chunks`, `chunk_index` is
1-based, `source_index` the line range (`start-end`), image name or 0-based PDF page,
`status` `pending`, `done` or `failed`, and `usage` JSON token counts
(`input_tokens`, `output_tokens`, `total_tokens`, `cached_tokens`,
`reasoning_tokens`, in synchronous and batch rows alike). In `ranges`,
`range_index` is 1-based, `original_start` and `original_end` are the starting bounds,
and `adjusted_start`, `adjusted_end` and `should_delete` (0 or 1) hold the boundary
decision of `ranges adjust` or `--chunking adjust`, NULL for a range without one. Open
the database read-only and close it after the query, so a running extraction never
waits on the reader:

```python
import sqlite3
from pathlib import Path

database = Path("out") / "chronominer.sqlite"
connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
try:
    rows = connection.execute(
        "SELECT chunk_index, entry_index, item, price FROM entries "
        "WHERE document = ? ORDER BY chunk_index, entry_index",
        ("list",),
    ).fetchall()
finally:
    connection.close()
```

If a read fails with "attempt to write a readonly database" after an interrupted run,
run `chronominer render --input <folder> --dry-run` on the output folder to roll back
the journal.

`render` rewrites the files of every document in the databases at or below `--input`
without model calls, in the formats each document was rendered in unless `--formats`
names others. A file that matches neither its recorded hash nor the new rendering was
edited and is kept unless `--force` is given; a file that cannot be replaced is marked
stale while its units stay stored (exit 1).

A database without schema version is not a ChronoMiner store; it, a newer one and an
unreadable one exit 2 in every command and are left as they are.

`drop` deletes the selected documents of one store with their chunks, line ranges,
responses, entries and run metadata, so the next extraction processes them from the
start; runs and batch submissions stay recorded. It deletes a document's rendered
files (`_output.json`, the CSV and a `_line_ranges.txt` the store rendered) whose
contents still match a recorded hash. A file changed outside the tool, missing,
unreadable or still recorded for a document that stays is kept and named; a kept
`_output.json`, or a kept CSV in the formats of a run, makes that run list the item
as `unstored` and skip it until the file is moved away. A document
that another live run leases, or whose chunks a submitting, submitted or unresolved
batch holds, is refused and left as it is (exit 1); the others are dropped.
`--dry-run` lists the same outcome without changing anything.

## Batch Processing

`--batch` submits the chunks of an extraction to the batch API of OpenAI, Anthropic or
Google; OpenRouter and custom endpoints have none. Requests are split by the
provider's count and byte limits, and each part becomes one submission in the
database, committed before the upload.

`batch check` finds the databases below `--input`, finalizes every finished batch
into the store, renders the affected documents and deletes the batch's files at the
provider; failed requests become failed chunks for `repair`, and so do refusals,
content-filter stops and answers cut at the output-token limit. A submission whose
create got no confirmed response, or that a killed process left without a provider
batch ID, is unresolved, and its chunks are never resubmitted automatically. OpenAI
and Google batches name their submission and are adopted when exactly one matches.
Anthropic batches do not: `batch check` lists the candidates created around the
submission, and `--adopt BATCH_ID` or `--abandon SUBMISSION_ID` settles it.
`--abandon` also gives up a submitted batch whose results cannot be read (for example
one the provider cannot find); the batch is not cancelled, and a run submits its
chunks again.

`extract --batch` followed by `batch check --input DIR --wait` submits and collects in
one job. The wait checks again every `--poll-minutes` until no submission is pending
and starts no check after `--wait-hours`; a check that is running then finishes. A
submission stays pending while its batch runs and while another run holds one of its
documents. Failed and unresolved submissions do not keep the wait going, while a
provider error that is retried counts as pending. The limit is soft: OpenAI and
Anthropic expire a batch after 24 hours and Google after 48, a check finalizes an
expired batch as failed, and the wait then ends on its own; the limit bounds a wait
whose checks cannot read the batch statuses. The wait sleeps between checks without
holding a lease, so other runs can work the same folder, and each check records a run
in the database. It writes one line
before each sleep, such as `Waiting for batches: 4 pending; next check at 14:45
(check 3; 0.50 of 50 h).`, and one at the end, such as `Stopped after 22 checks and
5.25 h: settled.` Its counts cover every submission a check saw, read from the database
at the end; one a check reported as failed stays failed unless the database shows it
finalized. It exits 0 when nothing is pending and nothing failed, 1 when a submission
failed or waits unresolved, when `--wait-hours` passed with work pending or when a
check failed with an error that is not retried, 2 for a refused database, and 130 on
Ctrl+C, after writing the counts so far.
`batch cancel` cancels batches by ID (the provider is inferred from the ID), every
unfinished batch of each provider whose key is set (`--all`), or the open submissions
below a folder (`--input`), whose chunks return to pending.

## Repair

`repair --input PATH` takes a folder, a database or one file, whose document it finds
in the nearest `chronominer.sqlite` at or above it; a file no database records exits
2. It selects the failed chunks, narrowed by `--schema` and `--indices`, skips
chunks an open or unresolved batch holds, rebuilds each chunk from its source with the
image settings it was first rendered with, and extracts it with the schema and context
of its extraction run, synchronously or with `--batch`. Earlier responses stay stored,
and the document's files are rendered again. A document whose input changed since its
extraction is refused: its failed chunks stay unrepaired (exit 1), and `extract
--force` extracts it again.

## Resume

Resume is the default and reads only the store; a dry run shows each item's state.

| State | Meaning | Action |
| --- | --- | --- |
| `none` | no stored units | extract from the start |
| `partial` | chunks are pending or failed | extract those chunks |
| `complete` | no pending or failed chunks | skip; missing or stale files are rendered again without calls |
| `unstored` | a rendered file exists without store rows, for example after `drop` kept it | skip and list; `--force` extracts it again |
| `blocked` | another live run holds the item, or its chunks wait for a batch | skip; a held item makes the run exit 1 |
| `refused` | the input's content changed since its extraction, or units remain but the stored pages used other image settings or the done chunks another schema | not processed, counted as failed; `--force` starts it over |

A run leases the items it plans or processes, a complete item whose files would be
rendered again included. An interrupted `extract` run releases its leases, so a rerun
right away resumes with the chunks still to do. A run that is killed instead keeps
its leases until they expire (10 minutes); until then a rerun reports its items as
blocked and exits 1.

`--force` resets every stored unit of the selected items, also outside `--pages`, and
keeps the stored responses. A resume with another model proceeds with a warning, and
each unit records its model. A model of another family (OpenAI, Anthropic, Google)
sizes page images by other settings, so a partial image or PDF item resumed with it is
refused. A text item cut with `--chunking auto` keeps its stored
line ranges on resume, also under another model or `--chunk-tokens`. To resume, run
the same command again: the state lives in the database, not in the rendered files.

## Architecture

```text
chronominer/
  __main__.py   entry point: dispatch the subcommand
  spec.py       run specifications, the option tables and schema resolution
  settings.py   typed settings: machine values and the defaults block
  run.py        core: discover, plan, run
  events.py     run events reported to the front ends
  ext.py        extension seam with pass-through defaults
  providers.py  provider names and inference from model names
  discovery.py  input discovery, sidecar rules, type detection, output paths
  cli/          argparse from the option tables, reporter, --json, exit codes
  ui/           questionary wizard, review screen, rich progress
  common/       vendored modules shared with the sibling tools
  llm/          client factory, model calls, capabilities, prompts
  extract/      chunks, extraction calls, answer parsing, the unit pipeline
  ranges/       line-range format, generation, adjustment and their store
  batch/        submit, check, finalize, reconcile, cancel and repair
  output/       store views, resume classification, renderers
  resources/    prompts and the settings example
schemas/        example extraction schemas
```

The core imports nothing from `cli/` or `ui/`; it never prints, prompts or exits and
reports through run events. `common/` holds the tool-agnostic modules shared with
AutoExcerpter and ChronoTranscriber, vendored byte-identically: `common/MANIFEST.json`
pins each file by its SHA-256, and `tests/common/test_manifest.py` fails when a copy
drifts. ChronoMiner passes its own values to them: its image presets are
`IMAGE_PRESETS` in `spec.py`, its request rules `RULES` in `llm/options.py`, its
shared option rows come from `common/standard_options.py` and its settings blocks
from `common/settings_schema.py`. `ext.py` holds the extension hooks (extra
arguments, settings blocks, wizard steps, a startup hook, consent to batch
submission, the key variables of plan-time checks, client options and an attempt
context around every model call); its defaults pass everything through.

## Development

```bash
uv sync --all-extras
uv run pytest                        # hermetic suite
uv run ruff check .
uv run ruff format --check .
uv run mypy chronominer              # strict
```

The suite removes provider keys, points the home folder at a temporary directory,
blocks non-loopback connections and runs model calls against a scripted fake model and
fake batch services. After an intended change, rewrite the golden files with
`CM_UPDATE_GOLDEN=1` and review the diff; `tests/README.md` describes the layout.

Evaluation data for a subset of historical sources (ground-truth extractions and
benchmark runs) is available from the author upon request.

## Versioning

[CHANGELOG.md](CHANGELOG.md) describes each release.

## License

MIT License. Copyright (c) 2025 Paul Goetz. See [LICENSE](LICENSE) for details.
