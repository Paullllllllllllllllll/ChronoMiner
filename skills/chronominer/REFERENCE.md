# ChronoMiner reference

Disclosed reference for [SKILL.md](SKILL.md): what the README and `--help` leave out
about line ranges, page images and the store.

## Line ranges

`{stem}_line_ranges.txt` beside a text is an input the user may edit by hand. `ranges
adjust` and `extract --chunking adjust` rewrite it in place beside the input, also
when `--output` names another folder, and close the gaps between ranges, so every
line outside a deleted range stays in a chunk. A hand-edited file starts the stored
boundary decisions over.

`ranges generate` cuts by token count, so its starts fall inside entries: never
extract with `--chunking line-ranges` from ranges that `ranges adjust` has not decided.
After the adjustment, and before extracting, read the first non-empty line at every
adjusted start (`adjusted_start` in the `ranges` view; `units.outcome` says `moved`,
`kept`, `on_target`, `merged` or `deleted`) and confirm that it opens an entry of the
schema. A start inside a sentence, or on a method paragraph below a title, is a defect:
correct the adjust context and adjust again with `--force`.

- **Window.** The model sees `--context-window` lines (default 6) on each side of the
  start; the window doubles while it asks for more context, up to 64 times the first
  radius. For texts whose entries run past about 50 lines (narrative cookbooks, short
  printed lines), pass a wider `--context-window` (24) to save calls.
- **Merges.** A range whose entry start no window reaches, or whose nearest start lies
  past its end, joins the range before it (`merged`) when a scan of its interior
  agrees, so a chunk can span up to four ranges; a further merged range keeps its
  token cut and needs a wider `--context-window` or a hand edit.
- **Context.** A work's block takes precedence over the template only where the two
  conflict explicitly. When a generic head of the template does not hold for a work
  (a template that opens entries with an imperative paragraph, in a book whose
  entries open with a title and whose method paragraphs open with imperatives), say
  so in the work's block: name the lines that open an entry and the lines that never
  do.

## Page images

With no image value pinned in the settings, a new document takes the preset of its
model family, and a stored one resumes under the image settings it started with. An
image flag that differs from the stored settings refuses the document on resume, and
a partial image or PDF item cannot resume under a model of another family (OpenAI,
Anthropic, Google): finish it with its own family, or `--force`.

## Store

Query `chronominer.sqlite` through its views `entries`, `chunks` and `ranges`, opened
read-only (`?mode=ro` URI) and closed after the query, so a running extraction never
waits on the reader. README, Store, lists the columns and gives an example; the
tables behind the views may change. `render` rewrites `{stem}_output.json` and the
CSV from the store and keeps a `{stem}_output.json` into which another program wrote
metadata; run it without `--force` on such outputs.
