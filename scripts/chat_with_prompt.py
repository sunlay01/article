from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a system prompt from file and send one user query to the OpenAI Responses API."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="The user question to send. If omitted, stdin will be used.",
    )
    parser.add_argument(
        "--system-prompt",
        dest="system_prompt",
        help="Override the system prompt file path. Defaults to SYSTEM_PROMPT_PATH from .env.",
    )
    parser.add_argument(
        "--model",
        help="Override the model name. Defaults to OPENAI_MODEL from .env.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        help="Optional reasoning effort for supported models.",
    )
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_system_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"System prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def read_question(cli_question: str | None) -> str:
    if cli_question:
        return cli_question.strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    return input("Enter your question: ").strip()


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it first.")

    base_url = os.getenv("OPENAI_BASE_URL")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def extract_response_text(response: object) -> str:
    if isinstance(response, str):
        return response.strip()

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    payload = None
    if isinstance(response, dict):
        payload = response
    elif hasattr(response, "model_dump"):
        payload = response.model_dump()

    if payload is None:
        return str(response).strip()

    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()

    fragments: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                fragments.append(text.strip())
            elif isinstance(text, dict):
                value = text.get("value")
                if isinstance(value, str) and value.strip():
                    fragments.append(value.strip())

    return "\n".join(fragments).strip() or str(payload).strip()


def main() -> int:
    load_dotenv()
    args = parse_args()

    system_prompt_path = resolve_path(
        args.system_prompt or os.getenv("SYSTEM_PROMPT_PATH", "prompts/research_system_prompt_en.md")
    )
    system_prompt = read_system_prompt(system_prompt_path)
    question = read_question(args.question)
    if not question:
        raise RuntimeError("No question provided. Pass one on the command line or via stdin.")

    model = args.model or os.getenv("OPENAI_MODEL", "gpt-5.5")
    client = build_client()

    request_kwargs = {
        "model": model,
        "instructions": system_prompt,
        "input": question,
    }
    if args.reasoning_effort:
        request_kwargs["reasoning"] = {"effort": args.reasoning_effort}

    response = client.responses.create(**request_kwargs)
    print(extract_response_text(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
