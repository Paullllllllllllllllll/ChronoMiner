# Changelog

- **v1.11.0** (13 November 2026) -- When the attempt context of `ext.py` answers a
  failed call with the new decision `STOP`, because a provider will serve no
  further calls of the run, `extract`, `ranges adjust`, `repair` and the
  `classify` commands stop at once with the new exit code 3 instead of failing
  every remaining chunk. The calls already running finish, and the item in work
  keeps its finished chunks, is rendered and is released; its other chunks stay
  pending, and for `repair` these are the chunks it had reset, which an `extract`
  rerun takes up. No further item starts, one error line names the reason and
  what was left, the JSON lines of these commands gain a `stopped` field, null
  unless the run was stopped, and `--help` lists the exit code. The store's tool
  component moves to version 4, which indexes the entry answers that the
  staleness triggers read: extracting a document with many entry answers again
  no longer reads all of them at every entry write. An older ChronoMiner refuses
  a store at version 4. `ranges adjust` and `--chunking adjust` decide
  `--concurrency` line ranges at a time and commit them one after another.
  Planning classifies the items of an output root over one read connection. A
  render reads each document when its turn comes and leaves it alone, without a
  lease or a write, when its files and rows already match the store. The stored
  message of a failed attempt names the transport error behind it, for example
  `(cause: httpx.ReadTimeout)`. The shared package gains `Decision.STOP`,
  `RunStopped`, `StopSignal`, `heed_stop`, `ExitCode.STOPPED` and the
  connection-level `resume_state`.
- **v1.10.0** (12 November 2026) -- The context image is sent before every
  chunk to every model that takes images, whatever the provider, in synchronous
  runs and in batches; it was limited to OpenAI and OpenRouter in synchronous
  runs and to OpenAI in batches. A model without image input skips it with a
  warning in the log.
- **v1.9.1** (8 November 2026) -- pyasn1 moves from 0.6.4 to 0.6.5 in
  `uv.lock` for GHSA-2qm8-x737-gmw5, GHSA-9m75-vc3g-4x39, GHSA-fm9c-5rcf-2q5p
  and GHSA-mm57-4v3p-rchx. No source change.
- **v1.9.0** (8 November 2026) -- Native PDF renders now stay within the
  `max_pixels_per_page` memory guard (24 MP) like numeric renders and record
  `memory_guard` when it lowers the zoom. A shared `check_request_size` in
  `common/provider_requests.py` holds every request to its provider's limits: each
  image, the context image included, within the family's per-image payload limit, and a
  Google request's images and prompt text together within 20 MB; a request over a limit
  is not sent and fails its page or chunk with the reason, while the rest of a batch is
  submitted and requests within the limits are unchanged.
- **v1.8.0** (8 November 2026) -- Gemini models at `--image-detail original` now get
  their pages at the rendered size, without the 768x1536 box and its padding, and every
  image part of synchronous and batch requests carries media resolution `ultra_high`;
  before, Google dropped `original` and sent the box-fitted page at `high`. The shared
  `common/image_presets.py` sets the Google section accordingly and adds
  `native_render_unbounded`, which the local check of native renders now calls;
  ChronoTranscriber, ChronoMiner and AutoExcerpter carry the same copy. The warning that
  a resize profile discards native resolution no longer appears for pages sent unscaled.
- **v1.7.0** (7 November 2026) -- Refreshed dependencies within their current majors:
  `langchain-anthropic` 1.7.10, `langchain-core` 1.7.6, `langchain-google-genai` 4.4.3,
  `langchain-openai` 1.7.4 and `pydantic` 2.14.3 at runtime, and `coverage` 7.17.2,
  `pytest-asyncio` 1.4.1.post1, `ruff` 0.17.4 and `types-PyYAML` 6.0.12.20261103 in the
  dev extra, with each `>=` floor raised to match. No dependency was removed. Major
  upgrades of `anthropic` (2.x), `google-genai` (3.x), `openai` (3.x) and `PyMuPDF`
  (2.x) are held on the current majors.

- **v1.6.0** (6 November 2026) -- `extract`, `repair` and `ranges adjust` take
  `--presence-penalty`, `--frequency-penalty`, `--top-k`, `--min-p` and
  `--repetition-penalty`, settable from `defaults:` like `--temperature`. Unset, a
  request is unchanged. A set value goes to the providers that take it: OpenRouter
  and custom endpoints get all five, Google models that take samplers the penalties
  and `top_k`, Anthropic `top_k` when no thinking is sent; the OpenAI Responses API
  takes none of them. Elsewhere the run warns that the value has no effect, and the
  dry-run view drops it. The shared `common/` modules `provider_requests.py`,
  `spec_view.py` and `standard_options.py` are synced with AutoExcerpter and
  ChronoTranscriber.
- **v1.5.0** (6 November 2026) -- `extract`, `repair` and `ranges adjust` take
  `--openrouter-provider NAME`, which pins the provider that serves OpenRouter model
  requests: each request then carries OpenRouter's routing object `{"order":
  [NAME], "allow_fallbacks": false, "require_parameters": true}` as `provider`, so
  one provider serves the run and a request it cannot serve fails instead of going
  to another provider. The flag is set on the command line only; for other providers
  it has no effect and says so, and the dry-run spec shows it for OpenRouter models.
  Decision-model requests keep OpenRouter's routing. The shared `common/` modules
  `provider_requests.py`, `spec_view.py` and `standard_options.py` are synced
  with AutoExcerpter and ChronoTranscriber.
- **v1.4.2** (6 November 2026) -- Discovery no longer enters `.mabillon` folders,
  in which Mabillon v0.10.0 keeps its working state below an output root it shares
  with ChronoMiner. A recursive extraction over such a root used to take the
  agent's unit files, notes and converted text as inputs.
- **v1.4.1** (5 November 2026) -- A rerun of `extract` after a killed run no longer
  skips the killed run's items for the 10 minutes its leases stay valid. Before the
  run it reads the leases on its items again every 90 seconds; a lease whose expiry
  does not move belongs to a run that has stopped, and the rerun waits until it
  expires and then extracts the item, while an item whose lease is renewed stays
  blocked and exits 1. A leased document without stored chunks is now blocked at
  planning.
- **v1.4.0** (1 November 2026) -- Line-range adjustment keeps boundaries on entry
  starts. While the model asks for more context, the window doubles, up to 64 times its
  first radius on each side. A range whose entry start lies beyond the last window or
  past its end, and whose interior holds no start, is merged into the range before it
  (unit outcome `merged`, at most three per range) instead of keeping a cut inside the
  entry. Merges and deletions wait until every range holds a decision, and a marker
  that matches several lines resolves to a line it begins. The new `--marker-length
  MIN-MAX` (default 5-15) of `ranges adjust` and `extract --chunking adjust` sets the
  marker length the prompt asks for, and shorter markers are rejected and asked for
  again. A heartbeat renews the document leases while ranges are adjusted.
- **v1.3.0** (1 November 2026) -- Refresh the dependencies within their current majors
  and cap `anthropic`, `openai` and PyMuPDF below their next majors. PDF code imports
  `pymupdf` instead of the deprecated `fitz`, whose warning broke the interrupt and
  killed-submit handshakes. A new `error_type_of` in `common/capture.py` keeps stored
  error types at their provider names under the LangChain 1.7 error wrappers, so a
  server error is still recorded as `InternalServerError`. The shared capability table
  adds `gemini-4-argon` and corrects the context limit of `gemini-3.8-flash` and
  `gemini-3.7-flash` to 1,048,576 tokens.
- **v1.2.2** (30 October 2026) -- A complete document no longer takes a
  requested file the store holds no record of as current: such a file is
  rendered, so the overwrite guard records it when it matches the rendering and
  otherwise keeps it with a warning. `ranges generate` fails for a text that
  yields no line range and keeps its stored ranges and file; earlier it deleted
  the stored ranges, wrote no file and reported success. The vendored common
  package matches ChronoTranscriber v1.2.3.
- **v1.2.1** (30 October 2026) -- The vendored common package matches
  ChronoTranscriber v1.2.2: a write that creates, writes, resets or deletes a
  document's units marks every rendered file of the document stale, the
  line-ranges file included, and rendering clears the flag of each file it
  records. A format left out of the run that completed a document, such as the
  CSV after a `--formats json` resume of a partial document, is therefore
  rendered again by the next run that asks for it; earlier it kept the rows of
  the partial run.
- **v1.2.0** (30 October 2026) -- `chronominer classify chunks`, `classify entries`
  and `classify documents` ask a decision model the questions of a classifier file
  about the extraction chunks of input items, the entries of finished documents, or
  each item's leading chunks or first pages, and record the answers in the new
  `classifications` view of `chronominer.sqlite` without changing any extraction.
  Classifiers resolve like schemas from the new `classifiers/` folder, which ships
  `chunk_kind`, `language`, `document_type` and six entry classifiers for the
  example schemas. Chunks are cut as `extract` cuts them without writing to the
  extraction: a stored document keeps its stored line ranges, and `--chunking
  adjust` is refused. Answers with an unchanged fingerprint, input and content are
  reused and recorded again under the run unless `--force`, an item whose stored
  input changed is refused, and entry answers are written under the document's
  lease. The vendored common package matches ChronoTranscriber's shared core:
  `classifiers.py`, `decision_runner.py`, `classify_store.py` (the
  `classifications` component), `images.decision_jpeg`, unleased store writes, the
  classifier options, migrations refused under another run's live lease, and
  malformed decision answers that keep their billed usage. The relevance pre-pass
  now asks through the shared runner, so such usage counts toward the run's totals.
  The tool's store component steps to version 3, whose triggers flag the answers
  about a chunk's entries stale whenever any writer replaces, resets or deletes the
  chunk, and a document's chunk answers when its stored chunk layout changes; an
  older ChronoMiner refuses the stepped store.
- **v1.1.2** (29 October 2026) -- The Pillow requirement is raised to
  13.0, the first release that fixes the FlashPix, TIFF-writer and
  multi-frame loader advisories, so that no install resolves a vulnerable
  Pillow. The lock already held 13.0.0.
- **v1.1.1** (29 October 2026) -- The guided UI asks for the decision-model
  pre-pass of a new extraction after the page images: off, shadow or apply,
  and with the pass on, the decision model, its API and its endpoint. The
  printed command carries the answers as `--decisions`, `--decision-model`,
  `--decision-provider` and `--decision-endpoint`.
- **v1.1.0** (29 October 2026) -- `extract --decisions shadow|apply` adds a
  decision-model pre-pass: OpenAI's `/v1/decisions` or OpenRouter's decision API
  (default `typesafe/jev-1.14`) answers per chunk still to extract whether its text
  or page image holds an entry of the schema's kind, built from the schema's name
  and description or its `decision_question` key. The relevance lands in the new
  `chunk_decisions` table and the `decision_probability` column of `chunks`, which
  also gains `outcome` (store component version 2). `apply` skips chunks below a
  threshold with the outcome `skipped_irrelevant`, but only where
  `decisions.thresholds` holds one for the run's request fingerprint;
  `repair --skipped` extracts such chunks. A skipped chunk is stored and rendered
  as a no-content answer. A pre-pass that fails leaves every chunk to extraction,
  and decision retries keep the timeout and validation budgets. The vendored
  common package matches ChronoTranscriber v1.1.1: it gains `decisions.py`
  (request bodies and typed answers for both dialects), `decision_client.py`,
  `page_types.py`, the shared decision-model options and the `decisions` settings
  schema.
- **v1.0.4** (27 October 2026) -- The capability table knows `gpt-6.1-terra`:
  it takes text and image input and structured outputs like `gpt-6.1-sol`,
  accepts the reasoning efforts none to max, and shares the GPT-6 handling of
  verbosity, Flex and prompt caching. Before, the model resolved as text-only
  and image extraction refused it. The vendored `chronominer/common/` modules
  now meet the Google docstring rules as well, so ruff no longer exempts them;
  the copies stay byte-identical with AutoExcerpter and ChronoTranscriber.
- **v1.0.3** (27 October 2026) -- Documentation patch: ruff now checks
  docstrings against the Google convention (pydocstyle rules `D`, without D105
  and D107; tests, notebooks, scratch and the vendored `chronominer/common/`
  are exempt). The protocol members of the batch, event and extraction
  interfaces gained docstrings, summaries that ran over two lines were split
  into a one-line summary and a body, and the prompt helpers in
  `chronominer/llm/prompt_utils.py` moved from NumPy-style sections to Google
  prose. Behavior is unchanged.
- **v1.0.2** (26 October 2026) -- Security patch: the lockfile moves Pillow to
  13.0.0, which closes the Pillow 12.3.0 advisories, among them the TIFF decoder
  reads GHSA-c6hw-5grf-ff96 and GHSA-q45j-58j2-45rg and the MIC frame-count loop
  GHSA-22hg-r758-rrw9, which scanned page images can reach. Dependency versions
  only; no code changes.
- **v1.0.1** (24 October 2026) -- Security patch: the lockfile moves certifi to
  2026.10.23, which closes GHSA-6q48-6c7w-9gw2, and urllib3 to 2.8.0, which closes
  GHSA-8988-9cw3-xx77, GHSA-vxq7-64xx-v4gw and GHSA-gh4c-6fx4-qh6g. Dependency versions
  only; no code changes.
- **v1.0.0** (19 October 2026) -- First release. The `chronominer` package has one
  core, `run.py`, with three entry points, `discover`, `plan` and `run` (plus
  `plan_ranges` and `run_ranges` for line ranges); the command line in `cli/` and the
  guided terminal UI in `ui/` both call them, and the core never prints, prompts or
  exits but reports through run events. `spec.py` declares one option table per
  command (`extract`, `ranges generate`, `ranges adjust`, `batch check`,
  `batch cancel`, `repair`, `render`, `drop`), from which the parser, the wizard and
  the help texts are built, so flags and wizard answers yield the same `RunSpec`.
  `settings.py` reads `config/settings.yaml`, a file named by `--settings` or the
  packaged example; its optional `defaults:` block sets run options below the flags.
  `discovery.py` finds the text, image and PDF items of an input, applies `--all`,
  `--select` and `--files`, resolves the `_extract_context` and `_adjust_context`
  sidecars and lays out the output paths. Texts are cut into chunks by token count
  (`auto`), by an editable `{stem}_line_ranges.txt` (`line-ranges`), or by line
  ranges whose boundaries the model first moves to entry boundaries (`adjust`);
  `ranges generate` and `ranges adjust` write that file. A JSON schema from
  `schemas/` and the three prompts in `resources/prompts/` shape each request, and
  `extract/` sends the chunks and parses their entries. Each output folder holds
  `chronominer.sqlite`, which records the runs, every provider attempt verbatim, the
  chunk units and the entries, so an interrupted run resumes from it. Its views
  `entries`, `chunks` and `ranges` are the read contract; leases keep a second run off
  a document in use, `render` rewrites `{stem}_output.json` and the CSV without model
  calls, `drop` deletes stored documents, and `repair` extracts failed chunks again.
  `batch/` submits, checks, waits for, finalizes and cancels batches over one backend
  each for OpenAI, Anthropic and Google. The logic shared with AutoExcerpter and
  ChronoTranscriber is vendored byte-identically in `common/` and pinned by
  `MANIFEST.json`: the run store and SQLite writer (`store.py`, `sqlite.py`), the
  request builder and options (`provider_requests.py`, `request_options.py`), the
  model capability table (`capability_table.py`, `capability_facts.py`,
  `capabilities.py`), image presets and payloads (`image_presets.py`, `images.py`),
  retry and rate limits (`retry.py`, `rate_limit.py`), option tables and the settings
  schema (`options.py`, `standard_options.py`, `settings_schema.py`), the wizard
  (`wizard.py`), the batch backends (`batch_openai.py`, `batch_anthropic.py`,
  `batch_google.py`) and the exit codes and JSON summary (`contract.py`). `ext.py` is
  the extension seam for a local access layer: its hooks `register_args`,
  `settings_blocks`, `wizard_steps`, `startup`, `batch_allowed`, `extra_argv`,
  `key_variables`, `describe_access`, `client_options` and the `attempt` context
  around every model call, whose roles are `extract` and `adjust`, pass everything
  through by default.
