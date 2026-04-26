from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from run_survey import OUTPUT_ROOT, normalize_venues, slugify_run


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUN_SURVEY_PATH = SCRIPT_DIR / "run_survey.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the literature survey pipeline from a JSON config file."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON config file that defines one or more survey jobs.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_years(raw_years: Any) -> list[int]:
    if isinstance(raw_years, int):
        return [raw_years]
    if not isinstance(raw_years, list) or not raw_years:
        raise ValueError("Each pipeline job must define a non-empty years list.")
    return sorted({int(item) for item in raw_years}, reverse=True)


def normalize_job(raw_job: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    job = {**defaults, **raw_job}
    venues_raw = job.get("venues") or job.get("venue")
    if isinstance(venues_raw, str):
        venues_raw = [venues_raw]
    if not venues_raw:
        raise ValueError("Each pipeline job must define venue or venues.")

    years = normalize_years(job.get("years"))
    venues = normalize_venues(list(venues_raw))
    mode = str(job.get("mode", "all"))
    if mode not in {"all", "scrape", "classify", "visualize"}:
        raise ValueError(f"Unsupported mode: {mode}")

    run_name = slugify_run(venues, years)
    name = str(job.get("name") or run_name)
    return {
        "name": name,
        "run_name": run_name,
        "mode": mode,
        "venues": venues,
        "years": years,
        "max_papers_per_venue_year": job.get("max_papers_per_venue_year"),
        "workers": int(job.get("workers", 6)),
        "batch_size": int(job.get("batch_size", 20)),
        "classify_workers": int(job.get("classify_workers", 2)),
        "top_k_topics": int(job.get("top_k_topics", 12)),
        "classify_model": job.get("classify_model"),
        "reasoning_effort": job.get("reasoning_effort"),
        "system_prompt": job.get("system_prompt"),
    }


def extract_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = config.get("defaults", {})
    if "jobs" in config:
        jobs = config["jobs"]
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("jobs must be a non-empty list.")
        return [normalize_job(job, defaults) for job in jobs]

    return [normalize_job(config, defaults)]


def build_job_command(job: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        str(RUN_SURVEY_PATH),
        "--mode",
        job["mode"],
        "--venues",
        *job["venues"],
        "--years",
        *[str(item) for item in job["years"]],
        "--workers",
        str(job["workers"]),
        "--batch-size",
        str(job["batch_size"]),
        "--classify-workers",
        str(job["classify_workers"]),
        "--top-k-topics",
        str(job["top_k_topics"]),
    ]
    if job["max_papers_per_venue_year"] is not None:
        command.extend(["--max-papers-per-venue-year", str(job["max_papers_per_venue_year"])])
    if job["classify_model"]:
        command.extend(["--classify-model", str(job["classify_model"])])
    if job["reasoning_effort"]:
        command.extend(["--reasoning-effort", str(job["reasoning_effort"])])
    if job["system_prompt"]:
        command.extend(["--system-prompt", str(job["system_prompt"])])
    return command


def build_initial_status(
    *,
    pipeline_name: str,
    config_path: Path,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "pipeline_name": pipeline_name,
        "config_path": str(config_path),
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "jobs": [
            {
                "name": job["name"],
                "run_name": job["run_name"],
                "mode": job["mode"],
                "venues": job["venues"],
                "years": job["years"],
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "duration_seconds": None,
                "output_dir": str(OUTPUT_ROOT / job["run_name"]),
                "log_path": None,
            }
            for job in jobs
        ],
    }


def main() -> int:
    load_dotenv()
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config = read_json(config_path)
    jobs = extract_jobs(config)
    continue_on_error = bool(config.get("continue_on_error", False))
    sleep_seconds_between_jobs = int(config.get("sleep_seconds_between_jobs", 0))
    pipeline_name = str(config.get("pipeline_name") or config_path.stem)
    pipeline_dir = ensure_dir(OUTPUT_ROOT / "pipelines" / pipeline_name)
    status_path = pipeline_dir / "pipeline_status.json"

    status = build_initial_status(pipeline_name=pipeline_name, config_path=config_path, jobs=jobs)
    status["status"] = "running"
    status["started_at"] = utc_now()
    atomic_write_json(status_path, status)

    overall_failed = False
    for index, job in enumerate(jobs):
        job_state = status["jobs"][index]
        job_state["status"] = "running"
        job_state["started_at"] = utc_now()
        log_path = pipeline_dir / f"{index + 1:02d}_{job['name']}.log"
        job_state["log_path"] = str(log_path)
        atomic_write_json(status_path, status)

        command = build_job_command(job)
        print(f"[pipeline] Starting job {index + 1}/{len(jobs)}: {job['name']}")
        print(f"[pipeline] Command: {' '.join(command)}")
        start_time = time.time()

        with log_path.open("w", encoding="utf-8", buffering=1) as handle:
            handle.write(f"$ {' '.join(command)}\n\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                handle.write(line)
                handle.flush()
            exit_code = process.wait()

        job_state["exit_code"] = exit_code
        job_state["finished_at"] = utc_now()
        job_state["duration_seconds"] = round(time.time() - start_time, 3)
        job_state["status"] = "completed" if exit_code == 0 else "failed"
        atomic_write_json(status_path, status)

        if exit_code != 0:
            overall_failed = True
            if not continue_on_error:
                break

        if sleep_seconds_between_jobs > 0 and index < len(jobs) - 1:
            time.sleep(sleep_seconds_between_jobs)

    status["finished_at"] = utc_now()
    status["status"] = "failed" if overall_failed else "completed"
    atomic_write_json(status_path, status)
    return 1 if overall_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
