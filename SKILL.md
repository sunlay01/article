---
name: ml-literature-survey
description: Use this skill when the user wants to scrape, classify, compare, or visualize machine learning paper trends across venues or years using title-and-abstract-only analysis, or when they want to extend the survey system to new venues via the venue registry and adapter framework.
---

# ML Literature Survey

Use this repo as the working skill when the task is about:

- running the `literature_survey` pipeline for one or more venues and years
- scheduling overnight survey jobs from a JSON config
- interpreting survey outputs such as `classified_papers.csv`, `progress.json`, frequency tables, and figures
- extending the scraper to new venues or source families

## Quick Start

- For one-off runs, use [literature_survey/run_survey.py](literature_survey/run_survey.py).
- For unattended multi-job runs, use [literature_survey/run_pipeline.py](literature_survey/run_pipeline.py) with a JSON config.
- Check [literature_survey/pipeline_config.example.json](literature_survey/pipeline_config.example.json) before creating a new pipeline config.

## Required Context

- Classification uses the model API and only sends `title + abstract`.
- Scraping uses source-specific adapters and does not read full papers.
- The venue registry lives in [literature_survey/venues.json](literature_survey/venues.json).

## Workflow

1. Confirm `.env` has the model credentials and base URL.
2. If the user wants an overnight run, create or update a JSON config and call `run_pipeline.py`.
3. If the user wants a single venue/year analysis, call `run_survey.py` directly.
4. During long classification runs, inspect:
   - `classified_papers.partial.csv`
   - `progress.json`
   - pipeline logs under `literature_survey/output/pipelines/`
5. After completion, summarize:
   - top topics
   - year-over-year shifts
   - any scraper or classification caveats

## Extending Venues

- If a new venue fits an existing source family, prefer editing [literature_survey/venues.json](literature_survey/venues.json).
- If a new source family is needed, add a new adapter in [literature_survey/run_survey.py](literature_survey/run_survey.py) and register it in `ADAPTERS`.

## Read Next When Needed

- Read [literature_survey/README.md](literature_survey/README.md) for runtime options, output files, and pipeline usage.
- Read [literature_survey/tag_taxonomy.json](literature_survey/tag_taxonomy.json) when changing the topic system.
