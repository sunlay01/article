from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUN_PIPELINE_PATH = SCRIPT_DIR / "run_pipeline.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for the active NeurIPS job to finish, then resume the repaired pipeline."
    )
    parser.add_argument("--pipeline-pid", type=int, required=True)
    parser.add_argument("--active-job-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def child_pids(pid: int) -> list[int]:
    result = subprocess.run(
        ["pgrep", "-P", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [int(line) for line in result.stdout.splitlines() if line.strip()]


def terminate_process_tree(pid: int, sig: int) -> None:
    for child_pid in child_pids(pid):
        terminate_process_tree(child_pid, sig)
    if pid_exists(pid):
        os.kill(pid, sig)


def write_resume_config(path: Path) -> None:
    payload = {
        "pipeline_name": "top4_venues_2023_2025_repair_resume",
        "continue_on_error": False,
        "sleep_seconds_between_jobs": 0,
        "defaults": {
            "workers": 6,
            "batch_size": 20,
            "classify_workers": 2,
            "top_k_topics": 12,
        },
        "jobs": [
            {
                "name": "icml_2023_2025_classify",
                "mode": "classify",
                "venues": ["icml"],
                "years": [2025, 2024, 2023],
            },
            {
                "name": "icml_2023_2025_visualize",
                "mode": "visualize",
                "venues": ["icml"],
                "years": [2025, 2024, 2023],
            },
            {
                "name": "neurips_2023_2025_classify",
                "mode": "classify",
                "venues": ["neurips"],
                "years": [2025, 2024, 2023],
            },
            {
                "name": "neurips_2023_2025_visualize",
                "mode": "visualize",
                "venues": ["neurips"],
                "years": [2025, 2024, 2023],
            },
            {
                "name": "cvpr_2023_2025",
                "mode": "all",
                "venues": ["cvpr"],
                "years": [2025, 2024, 2023],
            },
            {
                "name": "acl_2023_2025",
                "mode": "all",
                "venues": ["acl"],
                "years": [2025, 2024, 2023],
            },
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stop_old_pipeline(pipeline_pid: int) -> None:
    if not pid_exists(pipeline_pid):
        print(f"[supervisor] pipeline pid {pipeline_pid} already exited.")
        return

    print(f"[supervisor] stopping old pipeline pid {pipeline_pid}.")
    terminate_process_tree(pipeline_pid, signal.SIGTERM)

    deadline = time.time() + 15
    while time.time() < deadline:
        if not pid_exists(pipeline_pid):
            return
        time.sleep(1)

    if pid_exists(pipeline_pid):
        print(f"[supervisor] force killing old pipeline pid {pipeline_pid}.")
        terminate_process_tree(pipeline_pid, signal.SIGKILL)


def main() -> int:
    args = parse_args()

    print(
        f"[supervisor] watching active job pid {args.active_job_pid}; "
        f"pipeline pid {args.pipeline_pid}."
    )
    while pid_exists(args.active_job_pid):
        time.sleep(args.poll_seconds)

    print(f"[supervisor] active job pid {args.active_job_pid} exited.")
    stop_old_pipeline(args.pipeline_pid)

    config_path = SCRIPT_DIR / "pipeline_config_top4_2023_2025_repair_resume.json"
    write_resume_config(config_path)
    print(f"[supervisor] wrote resume config to {config_path}.")

    command = [sys.executable, "-u", str(RUN_PIPELINE_PATH), "--config", str(config_path)]
    print(f"[supervisor] starting repaired pipeline: {' '.join(command)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    print(f"[supervisor] repaired pipeline finished with exit code {result.returncode}.")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
