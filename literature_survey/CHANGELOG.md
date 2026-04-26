# Changelog

## 2026-04-26

- Added recursive classification fallback for malformed or unstable model batch outputs.
- When a batch fails validation or parsing, the classifier now splits the batch into smaller sub-batches until it succeeds or reaches a single paper.
- Added explicit fallback trace lines to pipeline job logs so batch-splitting behavior is visible during overnight runs.
- Documented the fallback behavior in `literature_survey/README.md`.
