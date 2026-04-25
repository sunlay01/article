from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import matplotlib
import pandas as pd
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI, RateLimitError
from tqdm import tqdm
from venue_registry import VenueSpec, load_venue_specs

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "research_system_prompt_en.md"
TAG_TAXONOMY_PATH = SCRIPT_DIR / "tag_taxonomy.json"
VENUE_REGISTRY_PATH = SCRIPT_DIR / "venues.json"
CACHE_DIR = SCRIPT_DIR / "cache"
OUTPUT_ROOT = SCRIPT_DIR / "output"

DEFAULT_YEARS = [2025, 2024, 2023]
USER_AGENT = "ai_learn-literature-survey/0.1"
PAPER_TYPES = [
    "method",
    "benchmark_dataset",
    "evaluation_analysis",
    "theory",
    "system",
    "application",
]

PREFERRED_CJK_FONTS = [
    "Hiragino Sans GB",
    "PingFang SC",
    "Songti SC",
    "STHeiti",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "SimHei",
]

THREAD_LOCAL = threading.local()
DEFAULT_HTTP_POOL_SIZE = 32

VENUE_SPECS = load_venue_specs(VENUE_REGISTRY_PATH)
DEFAULT_VENUES = list(VENUE_SPECS.keys())
VENUE_DISPLAY_NAMES = {key: spec.display_name for key, spec in VENUE_SPECS.items()}
VENUE_ALIASES = {
    alias.lower(): key
    for key, spec in VENUE_SPECS.items()
    for alias in (*spec.aliases, key)
}


@dataclass(frozen=True)
class TopicSpec:
    tag: str
    label: str
    zh_label: str
    description: str


class VenueAdapter:
    name = "base"

    def scrape(
        self,
        spec: VenueSpec,
        year: int,
        max_papers: int | None,
        workers: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class ICLROpenReviewAdapter(VenueAdapter):
    name = "iclr_openreview"

    def scrape(
        self,
        spec: VenueSpec,
        year: int,
        max_papers: int | None,
        workers: int,
    ) -> list[dict[str, Any]]:
        del workers
        return scrape_iclr_year(spec, year, max_papers)


class OpenReviewVenueAdapter(VenueAdapter):
    name = "openreview_venue"

    def scrape(
        self,
        spec: VenueSpec,
        year: int,
        max_papers: int | None,
        workers: int,
    ) -> list[dict[str, Any]]:
        del workers
        return scrape_openreview_venue(spec, year, max_papers)


class PMLRAdapter(VenueAdapter):
    name = "pmlr"

    def scrape(
        self,
        spec: VenueSpec,
        year: int,
        max_papers: int | None,
        workers: int,
    ) -> list[dict[str, Any]]:
        return scrape_icml_year(spec, year, max_papers, workers)


class CVFOpenAccessAdapter(VenueAdapter):
    name = "cvf_openaccess"

    def scrape(
        self,
        spec: VenueSpec,
        year: int,
        max_papers: int | None,
        workers: int,
    ) -> list[dict[str, Any]]:
        return scrape_cvpr_year(spec, year, max_papers, workers)


class ACLAnthologyAdapter(VenueAdapter):
    name = "acl_anthology"

    def scrape(
        self,
        spec: VenueSpec,
        year: int,
        max_papers: int | None,
        workers: int,
    ) -> list[dict[str, Any]]:
        del workers
        return scrape_acl_year(spec, year, max_papers)


ADAPTERS: dict[str, VenueAdapter] = {
    adapter.name: adapter
    for adapter in [
        ICLROpenReviewAdapter(),
        OpenReviewVenueAdapter(),
        PMLRAdapter(),
        CVFOpenAccessAdapter(),
        ACLAnthologyAdapter(),
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape top ML venue papers from the last three years, classify each paper "
            "based on title and abstract only, and generate tag frequency tables and visualizations."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["all", "scrape", "classify", "visualize"],
        default="all",
        help="Pipeline stage to run. Defaults to the full pipeline.",
    )
    parser.add_argument(
        "--venues",
        nargs="+",
        default=DEFAULT_VENUES,
        help="Venues to include. Supported: iclr icml neurips nips cvpr acl",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help="Publication years to include. Defaults to 2025 2024 2023.",
    )
    parser.add_argument(
        "--max-papers-per-venue-year",
        type=int,
        help="Optional cap for each venue-year, useful for prototyping.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent workers for detail-page scraping.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="How many papers to send to the model per classification request.",
    )
    parser.add_argument(
        "--classify-workers",
        type=int,
        default=max(2, min(6, (os.cpu_count() or 4) // 3 or 1)),
        help="Concurrent workers for model-based classification. Defaults to a conservative network-bound setting.",
    )
    parser.add_argument(
        "--classify-model",
        help="Override OPENAI_CLASSIFY_MODEL for the survey classifier.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        help="Optional reasoning effort for supported models.",
    )
    parser.add_argument(
        "--system-prompt",
        help="Override the base system prompt file. Defaults to SYSTEM_PROMPT_PATH or the repo prompt.",
    )
    parser.add_argument(
        "--top-k-topics",
        type=int,
        default=12,
        help="How many topics to keep in top-level charts.",
    )
    return parser.parse_args()


def normalize_venues(raw_venues: list[str]) -> list[str]:
    normalized = []
    for venue in raw_venues:
        item = VENUE_ALIASES.get(venue.lower())
        if item is None:
            raise ValueError(f"Unsupported venue: {venue}")
        if item not in normalized:
            normalized.append(item)
    return normalized


def load_topics() -> list[TopicSpec]:
    payload = json.loads(TAG_TAXONOMY_PATH.read_text(encoding="utf-8"))
    return [TopicSpec(**item) for item in payload["topics"]]


def get_venue_spec(venue: str) -> VenueSpec:
    try:
        return VENUE_SPECS[venue]
    except KeyError as exc:
        raise ValueError(f"Unsupported venue: {venue}") from exc


def slugify_run(venues: list[str], years: list[int]) -> str:
    return f"{'-'.join(venues)}_{min(years)}-{max(years)}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_plot_fonts() -> None:
    available_fonts = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    fallback_fonts = [font for font in PREFERRED_CJK_FONTS if font in available_fonts]
    if fallback_fonts:
        matplotlib.rcParams["font.family"] = ["sans-serif"]
        matplotlib.rcParams["font.sans-serif"] = fallback_fonts
    matplotlib.rcParams["axes.unicode_minus"] = False


def get_system_prompt_path(arg_path: str | None) -> Path:
    if arg_path:
        path = Path(arg_path)
    else:
        env_path = os.getenv("SYSTEM_PROMPT_PATH")
        path = Path(env_path) if env_path else DEFAULT_SYSTEM_PROMPT_PATH
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT}


def get_request_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "request_session", None)
    if session is not None:
        return session

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=DEFAULT_HTTP_POOL_SIZE,
        pool_maxsize=DEFAULT_HTTP_POOL_SIZE,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(build_headers())
    THREAD_LOCAL.request_session = session
    return session


def request_text(url: str, params: dict[str, Any] | None = None, timeout: int = 60) -> str:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            response = get_request_session().get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 8))
    assert last_error is not None
    raise last_error


def request_json(url: str, params: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            response = get_request_session().get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 8))
    assert last_error is not None
    raise last_error


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def unwrap_openreview_field(content: dict[str, Any], key: str) -> Any:
    value = content.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def paper_digest(title: str, abstract: str) -> str:
    return hashlib.sha1(f"{title}\n{abstract}".encode("utf-8")).hexdigest()


def infer_openreview_track(content: dict[str, Any]) -> str:
    venue_text = clean_text(str(unwrap_openreview_field(content, "venue") or "")).lower()
    if "oral" in venue_text:
        return "oral"
    if "spotlight" in venue_text:
        return "spotlight"
    if "poster" in venue_text:
        return "poster"
    return "accepted"


def extract_openreview_decision(note: dict[str, Any]) -> str:
    replies = note.get("details", {}).get("directReplies", [])
    for reply in replies:
        content = reply.get("content", {})
        title = unwrap_openreview_field(content, "title")
        decision = unwrap_openreview_field(content, "decision")
        if title == "Paper Decision" or decision:
            return clean_text(str(decision or ""))
    return ""


def is_accept_decision(decision: str) -> bool:
    return clean_text(decision).lower().startswith("accept")


def track_from_decision(decision: str) -> str:
    normalized = clean_text(decision).lower()
    if "oral" in normalized:
        return "oral"
    if "spotlight" in normalized:
        return "spotlight"
    if "poster" in normalized:
        return "poster"
    if "top-5" in normalized:
        return "notable-top-5%"
    if "top-25" in normalized:
        return "notable-top-25%"
    return "accepted"


def chunked(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def scrape_iclr_year(spec: VenueSpec, year: int, max_papers: int | None) -> list[dict[str, Any]]:
    endpoint_map = spec.config.get("submission_endpoints", {})
    endpoint_config = endpoint_map.get(str(year)) or endpoint_map.get(year)
    if endpoint_config is None:
        raise ValueError(f"{spec.display_name} submission endpoint mapping is not defined for year {year}")

    base_url = endpoint_config["base_url"]
    invitation_name = endpoint_config["invitation_name"]
    offset = 0
    collected: list[dict[str, Any]] = []

    while True:
        params = {
            "invitation": f"ICLR.cc/{year}/Conference/-/{invitation_name}",
            "details": "directReplies",
            "limit": 1000,
            "offset": offset,
        }
        payload = request_json(base_url, params=params, timeout=90)
        notes = payload.get("notes", [])
        if not notes:
            break

        for note in notes:
            content = note.get("content", {})
            title = clean_text(str(unwrap_openreview_field(content, "title") or ""))
            abstract = clean_text(str(unwrap_openreview_field(content, "abstract") or ""))
            if not title or not abstract:
                continue

            decision = extract_openreview_decision(note)
            if not is_accept_decision(decision):
                continue

            note_id = note["id"]
            collected.append(
                {
                    "paper_uid": f"{spec.key}:{year}:{note_id}",
                    "source_id": note_id,
                    "venue": spec.key,
                    "year": year,
                    "track": track_from_decision(decision),
                    "title": title,
                    "abstract": abstract,
                    "source": "openreview",
                    "source_url": f"https://openreview.net/forum?id={note_id}",
                    "official_primary_area": clean_text(
                        str(unwrap_openreview_field(content, "primary_area") or "")
                    ),
                }
            )
            if max_papers and len(collected) >= max_papers:
                return collected

        offset += len(notes)

    return collected


def scrape_openreview_venue(
    spec: VenueSpec,
    year: int,
    max_papers: int | None,
) -> list[dict[str, Any]]:
    base_url = "https://api2.openreview.net/notes"
    collected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    query_variants: list[dict[str, str]] = []
    venue_template = spec.config.get("venue_id_template")
    if venue_template:
        query_variants.append({"venueid": venue_template.format(year=year)})

    for label in spec.config.get("accepted_labels", []):
        query_variants.append({"content.venue": f"{spec.display_name} {year} {label}"})

    published_label = f"Published as a conference paper at {spec.display_name} {year}"
    query_variants.append({"content.venue": published_label})

    for variant in query_variants:
        offset = 0
        while True:
            params = {
                **variant,
                "limit": 1000,
                "offset": offset,
            }
            try:
                payload = request_json(base_url, params=params, timeout=60)
            except requests.exceptions.HTTPError as exc:
                response = getattr(exc, "response", None)
                if response is not None and response.status_code == 400 and offset == 0:
                    break
                raise
            notes = payload.get("notes", [])
            if not notes:
                break

            for note in notes:
                note_id = note["id"]
                if note_id in seen_ids:
                    continue
                content = note.get("content", {})
                title = clean_text(str(unwrap_openreview_field(content, "title") or ""))
                abstract = clean_text(str(unwrap_openreview_field(content, "abstract") or ""))
                if not title or not abstract:
                    continue
                seen_ids.add(note_id)
                collected.append(
                    {
                        "paper_uid": f"{spec.key}:{year}:{note_id}",
                        "source_id": note_id,
                        "venue": spec.key,
                        "year": year,
                        "track": infer_openreview_track(content),
                        "title": title,
                        "abstract": abstract,
                        "source": "openreview",
                        "source_url": f"https://openreview.net/forum?id={note_id}",
                        "official_primary_area": clean_text(
                            str(unwrap_openreview_field(content, "primary_area") or "")
                        ),
                    }
                )
                if max_papers and len(collected) >= max_papers:
                    return collected

            offset += len(notes)

    return collected


def scrape_icml_year(spec: VenueSpec, year: int, max_papers: int | None, workers: int) -> list[dict[str, Any]]:
    volume_map = spec.config.get("volume_map", {})
    volume = volume_map.get(str(year)) or volume_map.get(year)
    if volume is None:
        raise ValueError(f"{spec.display_name} volume mapping is not defined for year {year}")

    volume_url = f"https://proceedings.mlr.press/v{volume}/"
    soup = BeautifulSoup(request_text(volume_url, timeout=60), "html.parser")
    paper_nodes = soup.select("div.paper")
    pending: list[dict[str, Any]] = []

    for node in paper_nodes:
        title_node = node.select_one("p.title")
        abs_link = None
        for link in node.select("p.links a"):
            if link.get_text(strip=True).lower() == "abs":
                abs_link = link.get("href")
                break
        if not title_node or not abs_link:
            continue
        title = clean_text(title_node.get_text(" ", strip=True))
        pending.append(
            {
                "paper_uid": f"{spec.key}:{year}:{Path(abs_link).stem}",
                "source_id": Path(abs_link).stem,
                "venue": spec.key,
                "year": year,
                "track": "main",
                "title": title,
                "detail_url": abs_link,
                "source": "pmlr",
                "source_url": abs_link,
                "official_primary_area": "",
            }
        )
        if max_papers and len(pending) >= max_papers:
            break

    def fetch_detail(item: dict[str, Any]) -> dict[str, Any] | None:
        detail_soup = BeautifulSoup(request_text(item["detail_url"], timeout=60), "html.parser")
        abstract_node = detail_soup.select_one("#abstract")
        if not abstract_node:
            return None
        enriched = dict(item)
        enriched["abstract"] = clean_text(abstract_node.get_text(" ", strip=True))
        return enriched

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_detail, item) for item in pending]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"{spec.display_name} {year}",
        ):
            result = future.result()
            if result and result.get("abstract"):
                results.append(result)

    return sorted(results, key=lambda item: item["paper_uid"])


def scrape_cvpr_year(spec: VenueSpec, year: int, max_papers: int | None, workers: int) -> list[dict[str, Any]]:
    index_url = spec.config["index_url_template"].format(year=year)
    soup = BeautifulSoup(request_text(index_url, timeout=60), "html.parser")
    pending: list[dict[str, Any]] = []

    for link in soup.select("dt.ptitle a"):
        href = link.get("href")
        title = clean_text(link.get_text(" ", strip=True))
        if not href or not title:
            continue
        detail_url = urljoin(index_url, href)
        source_id = Path(href).stem.replace("_paper", "")
        pending.append(
            {
                "paper_uid": f"{spec.key}:{year}:{source_id}",
                "source_id": source_id,
                "venue": spec.key,
                "year": year,
                "track": "main",
                "title": title,
                "detail_url": detail_url,
                "source": "cvf_openaccess",
                "source_url": detail_url,
                "official_primary_area": "",
            }
        )
        if max_papers and len(pending) >= max_papers:
            break

    def fetch_detail(item: dict[str, Any]) -> dict[str, Any] | None:
        detail_soup = BeautifulSoup(request_text(item["detail_url"], timeout=60), "html.parser")
        abstract_node = detail_soup.select_one("#abstract")
        if not abstract_node:
            return None
        enriched = dict(item)
        enriched["abstract"] = clean_text(abstract_node.get_text(" ", strip=True))
        return enriched

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_detail, item) for item in pending]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"{spec.display_name} {year}",
        ):
            result = future.result()
            if result and result.get("abstract"):
                results.append(result)

    return sorted(results, key=lambda item: item["paper_uid"])


def is_acl_main_volume(title: str) -> bool:
    title_lower = title.lower()
    if "annual meeting of the association for computational linguistics" not in title_lower:
        return False
    if "findings" in title_lower or "tutorial" in title_lower or "student research workshop" in title_lower:
        return False
    return True


def scrape_acl_year(spec: VenueSpec, year: int, max_papers: int | None) -> list[dict[str, Any]]:
    event_url = spec.config["event_url_template"].format(year=year)
    soup = BeautifulSoup(request_text(event_url, timeout=90), "html.parser")
    volume_links: dict[str, str] = {}

    for anchor in soup.select('a[href^="/volumes/"]'):
        href = anchor.get("href")
        title = clean_text(anchor.get_text(" ", strip=True))
        if not href or not re.match(rf"^/volumes/{year}\.acl[\w-]*/$", href):
            continue
        if not is_acl_main_volume(title):
            continue
        volume_links[href] = title

    results: list[dict[str, Any]] = []
    for href in tqdm(sorted(volume_links.keys()), desc=f"{spec.display_name} {year} volumes"):
        volume_url = urljoin("https://aclanthology.org/", href)
        volume_soup = BeautifulSoup(request_text(volume_url, timeout=90), "html.parser")
        for row in volume_soup.select("div.d-sm-flex.align-items-stretch.mb-3"):
            title_link = row.select_one("strong a")
            if not title_link:
                continue
            paper_href = title_link.get("href")
            title = clean_text(title_link.get_text(" ", strip=True))
            abstract_card = row.find_next_sibling("div", class_="card")
            abstract = clean_text(abstract_card.get_text(" ", strip=True)) if abstract_card else ""
            if not paper_href or not title or not abstract:
                continue
            if paper_href.endswith(".0/"):
                continue
            paper_id = paper_href.strip("/").replace("/", "")
            results.append(
                {
                    "paper_uid": f"{spec.key}:{year}:{paper_id}",
                    "source_id": paper_id,
                    "venue": spec.key,
                    "year": year,
                    "track": "main",
                    "title": title,
                    "abstract": abstract,
                    "source": "acl_anthology",
                    "source_url": urljoin("https://aclanthology.org/", paper_href),
                    "official_primary_area": "",
                }
            )
            if max_papers and len(results) >= max_papers:
                return results

    deduped = {item["paper_uid"]: item for item in results}
    return sorted(deduped.values(), key=lambda item: item["paper_uid"])


def scrape_venue_year(venue: str, year: int, max_papers: int | None, workers: int) -> list[dict[str, Any]]:
    spec = get_venue_spec(venue)
    try:
        adapter = ADAPTERS[spec.adapter]
    except KeyError as exc:
        raise ValueError(f"No adapter is registered for venue {venue}: {spec.adapter}") from exc
    return adapter.scrape(spec, year, max_papers, workers)


def scrape_all(
    venues: list[str],
    years: list[int],
    max_papers: int | None,
    workers: int,
) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    for venue in venues:
        for year in sorted(years, reverse=True):
            rows = scrape_venue_year(venue, year, max_papers, workers)
            print(f"Scraped {len(rows)} papers for {VENUE_DISPLAY_NAMES[venue]} {year}")
            all_rows.extend(rows)
    frame = pd.DataFrame(all_rows)
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(subset=["paper_uid"]).copy()
    frame["paper_digest"] = frame.apply(
        lambda row: paper_digest(str(row["title"]), str(row["abstract"])),
        axis=1,
    )
    return frame.sort_values(["venue", "year", "paper_uid"]).reset_index(drop=True)


def create_classifier_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it first.")

    base_url = os.getenv("OPENAI_BASE_URL")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def get_classifier_client() -> OpenAI:
    client = getattr(THREAD_LOCAL, "classifier_client", None)
    if client is not None:
        return client

    client = create_classifier_client()
    THREAD_LOCAL.classifier_client = client
    return client


def extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    payload: dict[str, Any] | None = None
    if isinstance(response, dict):
        payload = response
    elif hasattr(response, "model_dump"):
        payload = response.model_dump()

    if payload is None:
        return clean_text(str(response))

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

    if fragments:
        return "\n".join(fragments)

    return clean_text(str(payload))


def build_tag_instruction(topics: list[TopicSpec]) -> str:
    lines = [
        "You are doing topic tagging for a machine learning literature survey.",
        "Use only the paper title and abstract. Do not assume details from the full paper.",
        "Return valid JSON only.",
        "For each paper, output:",
        "- paper_uid",
        "- direction_summary_zh: a short Chinese summary of the paper's main direction, at most 18 Chinese characters if possible",
        "- primary_topic: exactly one tag from the allowed taxonomy",
        "- secondary_topics: zero to three distinct tags from the allowed taxonomy",
        "- paper_type: exactly one of method, benchmark_dataset, evaluation_analysis, theory, system, application",
        "- confidence: a float between 0 and 1",
        "Choose the primary topic as the paper's central research problem, not just a surface keyword.",
        "Use evaluation_benchmarks as the primary topic only when the benchmark or evaluation framework is the actual central contribution.",
        "Allowed topics:",
    ]
    for topic in topics:
        lines.append(f"- {topic.tag}: {topic.label} - {topic.description}")
    return "\n".join(lines)


def sanitize_model_input_text(text: str) -> str:
    sanitized = text
    replacements = [
        (r"(?i)jailbreak", "jail break"),
        (r"越狱", "越-狱"),
        (r"破限", "破-限"),
        (r"脱限", "脱-限"),
    ]
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized)
    return sanitized


def build_classification_input(
    batch: list[dict[str, Any]],
    *,
    sanitize_sensitive_terms: bool = False,
) -> str:
    records = []
    for item in batch:
        title = item["title"]
        abstract = item["abstract"]
        if sanitize_sensitive_terms:
            title = sanitize_model_input_text(title)
            abstract = sanitize_model_input_text(abstract)
        records.append(
            {
                "paper_uid": item["paper_uid"],
                "title": title,
                "abstract": abstract,
            }
        )
    return (
        "Classify the following papers based on title and abstract only.\n"
        "Return a JSON array with one object per paper.\n\n"
        f"{json.dumps(records, ensure_ascii=False, indent=2)}"
    )


def is_sensitive_word_error(exc: BadRequestError) -> bool:
    message = clean_text(str(exc)).lower()
    return "敏感词" in message or "jailbreak" in message or "越狱" in message or "破限" in message or "脱限" in message


def extract_json_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = None
    opener = None
    for index, char in enumerate(stripped):
        if char in "[{":
            start = index
            opener = char
            break
    if start is None or opener is None:
        raise ValueError("No JSON block found in model output.")

    closer = "]" if opener == "[" else "}"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    raise ValueError("Incomplete JSON block in model output.")


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        cache[record["paper_digest"]] = record
    return cache


def append_cache(cache_path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(cache_path.parent)
    with cache_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def atomic_write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)


def validate_classification(
    parsed: list[dict[str, Any]],
    batch: list[dict[str, Any]],
    allowed_tags: set[str],
) -> list[dict[str, Any]]:
    expected_ids = {item["paper_uid"] for item in batch}
    received_ids = {item.get("paper_uid") for item in parsed}
    if expected_ids != received_ids:
        raise ValueError("Model output does not match the requested paper IDs.")

    normalized: list[dict[str, Any]] = []
    for item in parsed:
        primary = item.get("primary_topic")
        secondaries = item.get("secondary_topics", [])
        if primary not in allowed_tags:
            primary = "other_misc"
        if not isinstance(secondaries, list):
            secondaries = []
        secondaries = [tag for tag in secondaries if tag in allowed_tags and tag != primary][:3]
        paper_type = item.get("paper_type")
        if paper_type not in PAPER_TYPES:
            paper_type = "method"
        try:
            confidence = float(item.get("confidence", 0.5))
        except Exception:  # noqa: BLE001
            confidence = 0.5
        confidence = min(max(confidence, 0.0), 1.0)
        normalized.append(
            {
                "paper_uid": item["paper_uid"],
                "direction_summary_zh": clean_text(str(item.get("direction_summary_zh", ""))) or "未明确分类",
                "primary_topic": primary,
                "secondary_topics": secondaries,
                "paper_type": paper_type,
                "confidence": round(confidence, 3),
            }
        )
    return normalized


def classify_batch(
    batch: list[dict[str, Any]],
    *,
    instructions: str,
    model: str,
    reasoning_effort: str | None,
    allowed_tags: set[str],
    digest_by_paper_uid: dict[str, str],
) -> list[dict[str, Any]]:
    plain_input = build_classification_input(batch)
    sanitized_input = build_classification_input(batch, sanitize_sensitive_terms=True)
    request_kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": plain_input,
    }
    if reasoning_effort:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    raw_text = ""
    client = get_classifier_client()
    for attempt in range(1, 6):
        try:
            response = client.responses.create(**request_kwargs)
        except RateLimitError:
            if attempt == 5:
                raise
            time.sleep(min(10 * attempt, 30))
            continue
        except BadRequestError as exc:
            if request_kwargs["input"] != sanitized_input and is_sensitive_word_error(exc):
                request_kwargs["input"] = sanitized_input
                continue
            raise
        raw_text = extract_response_text(response)
        try:
            parsed_block = extract_json_block(raw_text)
            parsed = json.loads(parsed_block)
            if isinstance(parsed, dict):
                parsed = parsed.get("papers", [])
            if not isinstance(parsed, list):
                raise ValueError("Expected a JSON array from the model.")
            validated = validate_classification(parsed, batch, allowed_tags)
            return [
                {
                    "paper_digest": digest_by_paper_uid[item["paper_uid"]],
                    **item,
                }
                for item in validated
            ]
        except Exception as exc:  # noqa: BLE001
            if attempt == 5:
                raise RuntimeError(
                    f"Failed to parse classification response after 5 attempts. Last output:\n{raw_text}"
                ) from exc
            time.sleep(1.5 * attempt)

    return []


def classify_papers(
    papers: pd.DataFrame,
    topics: list[TopicSpec],
    system_prompt_path: Path,
    model: str,
    reasoning_effort: str | None,
    batch_size: int,
    classify_workers: int,
    partial_path: Path | None = None,
    progress_path: Path | None = None,
) -> pd.DataFrame:
    if papers.empty:
        return papers

    cache_path = CACHE_DIR / "classification_cache.jsonl"
    cache = load_cache(cache_path)
    base_prompt = read_text(system_prompt_path)
    task_prompt = build_tag_instruction(topics)
    instructions = f"{base_prompt}\n\n# Additional Task\n{task_prompt}"
    allowed_tags = {topic.tag for topic in topics}

    cached_rows: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    for row in papers.to_dict("records"):
        cached = cache.get(row["paper_digest"])
        if cached:
            cached_rows.append(cached)
        else:
            pending_rows.append(row)

    print(f"Using {len(cached_rows)} cached classifications; {len(pending_rows)} papers still need API tagging.")

    effective_batch_size = max(batch_size, 1)
    new_cache_records: list[dict[str, Any]] = []
    digest_by_paper_uid = {row["paper_uid"]: row["paper_digest"] for row in pending_rows}
    batches = chunked(pending_rows, effective_batch_size)
    effective_workers = max(1, classify_workers)

    print(
        f"Classifying with batch_size={effective_batch_size}, "
        f"classify_workers={effective_workers}."
    )

    combined = {record["paper_digest"]: record for record in cached_rows}
    completed_batches = 0
    total_batches = len(batches)

    if partial_path and progress_path:
        write_classification_progress(
            papers=papers,
            classified_records_by_digest=combined,
            topics=topics,
            partial_path=partial_path,
            progress_path=progress_path,
            total_batches=total_batches,
            completed_batches=completed_batches,
            batch_size=effective_batch_size,
            classify_workers=effective_workers,
            status="running",
            cached_at_start=len(cached_rows),
        )

    if effective_workers == 1:
        for batch in tqdm(batches, desc="Classifying batches"):
            batch_cache_records = classify_batch(
                batch,
                instructions=instructions,
                model=model,
                reasoning_effort=reasoning_effort,
                allowed_tags=allowed_tags,
                digest_by_paper_uid=digest_by_paper_uid,
            )
            if batch_cache_records:
                append_cache(cache_path, batch_cache_records)
                new_cache_records.extend(batch_cache_records)
                combined.update({record["paper_digest"]: record for record in batch_cache_records})
            completed_batches += 1
            if partial_path and progress_path:
                write_classification_progress(
                    papers=papers,
                    classified_records_by_digest=combined,
                    topics=topics,
                    partial_path=partial_path,
                    progress_path=progress_path,
                    total_batches=total_batches,
                    completed_batches=completed_batches,
                    batch_size=effective_batch_size,
                    classify_workers=effective_workers,
                    status="running",
                    cached_at_start=len(cached_rows),
                )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(
                    classify_batch,
                    batch,
                    instructions=instructions,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    allowed_tags=allowed_tags,
                    digest_by_paper_uid=digest_by_paper_uid,
                )
                for batch in batches
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Classifying batches ({effective_workers} workers)",
            ):
                batch_cache_records = future.result()
                if batch_cache_records:
                    append_cache(cache_path, batch_cache_records)
                    new_cache_records.extend(batch_cache_records)
                    combined.update({record["paper_digest"]: record for record in batch_cache_records})
                completed_batches += 1
                if partial_path and progress_path:
                    write_classification_progress(
                        papers=papers,
                        classified_records_by_digest=combined,
                        topics=topics,
                        partial_path=partial_path,
                        progress_path=progress_path,
                        total_batches=total_batches,
                        completed_batches=completed_batches,
                        batch_size=effective_batch_size,
                        classify_workers=effective_workers,
                        status="running",
                        cached_at_start=len(cached_rows),
                    )

    enriched = build_enriched_classification_frame(papers, combined, topics)
    if partial_path and progress_path:
        write_classification_progress(
            papers=papers,
            classified_records_by_digest=combined,
            topics=topics,
            partial_path=partial_path,
            progress_path=progress_path,
            total_batches=total_batches,
            completed_batches=completed_batches,
            batch_size=effective_batch_size,
            classify_workers=effective_workers,
            status="completed",
            cached_at_start=len(cached_rows),
        )
    return enriched


def parse_secondary_topics(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def topic_en_label_map(topics: list[TopicSpec]) -> dict[str, str]:
    return {topic.tag: topic.label for topic in topics}


def topic_zh_label_map(topics: list[TopicSpec]) -> dict[str, str]:
    return {topic.tag: topic.zh_label for topic in topics}


def topic_bilingual_label_map(topics: list[TopicSpec]) -> dict[str, str]:
    return {topic.tag: f"{topic.label} / {topic.zh_label}" for topic in topics}


def topic_plot_label_map(topics: list[TopicSpec]) -> dict[str, str]:
    return {topic.tag: topic.zh_label for topic in topics}


def format_plot_label(label: str) -> str:
    return label.replace(" / ", "\n")


def build_enriched_classification_frame(
    papers: pd.DataFrame,
    classified_records_by_digest: dict[str, dict[str, Any]],
    topics: list[TopicSpec],
) -> pd.DataFrame:
    if papers.empty or not classified_records_by_digest:
        return papers.iloc[0:0].copy()

    enriched = papers[papers["paper_digest"].isin(classified_records_by_digest)].copy()
    if enriched.empty:
        return enriched

    en_label_map = topic_en_label_map(topics)
    zh_label_map = topic_zh_label_map(topics)
    bilingual_label_map = topic_bilingual_label_map(topics)

    enriched["direction_summary_zh"] = enriched["paper_digest"].map(
        lambda digest: classified_records_by_digest[digest]["direction_summary_zh"]
    )
    enriched["primary_topic"] = enriched["paper_digest"].map(
        lambda digest: classified_records_by_digest[digest]["primary_topic"]
    )
    enriched["secondary_topics"] = enriched["paper_digest"].map(
        lambda digest: json.dumps(
            classified_records_by_digest[digest]["secondary_topics"],
            ensure_ascii=False,
        )
    )
    enriched["paper_type"] = enriched["paper_digest"].map(
        lambda digest: classified_records_by_digest[digest]["paper_type"]
    )
    enriched["confidence"] = enriched["paper_digest"].map(
        lambda digest: classified_records_by_digest[digest]["confidence"]
    )
    enriched["primary_topic_label_en"] = enriched["primary_topic"].map(en_label_map)
    enriched["primary_topic_label_zh"] = enriched["primary_topic"].map(zh_label_map)
    enriched["primary_topic_label_bilingual"] = enriched["primary_topic"].map(bilingual_label_map)
    enriched["secondary_topic_labels_bilingual"] = enriched["secondary_topics"].apply(
        lambda value: json.dumps(
            [bilingual_label_map.get(tag, tag) for tag in parse_secondary_topics(value)],
            ensure_ascii=False,
        )
    )
    return enriched.sort_values(["venue", "year", "paper_uid"]).reset_index(drop=True)


def write_classification_progress(
    *,
    papers: pd.DataFrame,
    classified_records_by_digest: dict[str, dict[str, Any]],
    topics: list[TopicSpec],
    partial_path: Path,
    progress_path: Path,
    total_batches: int,
    completed_batches: int,
    batch_size: int,
    classify_workers: int,
    status: str,
    cached_at_start: int,
) -> None:
    partial_frame = build_enriched_classification_frame(papers, classified_records_by_digest, topics)
    atomic_write_csv(partial_path, partial_frame)

    total_papers = len(papers)
    classified_total = len(partial_frame)
    remaining = max(total_papers - classified_total, 0)
    progress_fraction = (classified_total / total_papers) if total_papers else 1.0
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_papers": total_papers,
        "classified_papers": classified_total,
        "remaining_papers": remaining,
        "progress_fraction": round(progress_fraction, 6),
        "progress_percent": round(progress_fraction * 100, 2),
        "cached_at_start": cached_at_start,
        "newly_classified_this_run": max(classified_total - cached_at_start, 0),
        "total_batches": total_batches,
        "completed_batches": completed_batches,
        "batch_size": batch_size,
        "classify_workers": classify_workers,
        "partial_csv_path": str(partial_path),
    }
    atomic_write_text(progress_path, json.dumps(payload, ensure_ascii=False, indent=2))


def write_frequency_tables(classified: pd.DataFrame, topics: list[TopicSpec], output_dir: Path) -> None:
    if classified.empty:
        return
    freq_dir = ensure_dir(output_dir / "frequency_tables")
    en_label_map = topic_en_label_map(topics)
    zh_label_map = topic_zh_label_map(topics)
    bilingual_label_map = topic_bilingual_label_map(topics)

    frame = classified.copy()
    frame["secondary_topics"] = frame["secondary_topics"].apply(parse_secondary_topics)
    frame["all_topics"] = frame.apply(
        lambda row: [row["primary_topic"]] + [tag for tag in row["secondary_topics"] if tag != row["primary_topic"]],
        axis=1,
    )

    primary_counts = (
        frame.groupby(["venue", "primary_topic"])
        .size()
        .reset_index(name="count")
        .sort_values(["venue", "count", "primary_topic"], ascending=[True, False, True])
    )
    primary_counts["topic_label_en"] = primary_counts["primary_topic"].map(en_label_map)
    primary_counts["topic_label_zh"] = primary_counts["primary_topic"].map(zh_label_map)
    primary_counts["topic_label_bilingual"] = primary_counts["primary_topic"].map(bilingual_label_map)

    exploded_all = frame.explode("all_topics")
    all_counts = (
        exploded_all.groupby(["venue", "all_topics"])
        .size()
        .reset_index(name="count")
        .rename(columns={"all_topics": "topic"})
        .sort_values(["venue", "count", "topic"], ascending=[True, False, True])
    )
    all_counts["topic_label_en"] = all_counts["topic"].map(en_label_map)
    all_counts["topic_label_zh"] = all_counts["topic"].map(zh_label_map)
    all_counts["topic_label_bilingual"] = all_counts["topic"].map(bilingual_label_map)

    year_counts = (
        frame.groupby(["year", "primary_topic"])
        .size()
        .reset_index(name="count")
        .sort_values(["year", "count", "primary_topic"], ascending=[True, False, True])
    )
    year_counts["topic_label_en"] = year_counts["primary_topic"].map(en_label_map)
    year_counts["topic_label_zh"] = year_counts["primary_topic"].map(zh_label_map)
    year_counts["topic_label_bilingual"] = year_counts["primary_topic"].map(bilingual_label_map)

    venue_year_counts = (
        frame.groupby(["venue", "year", "primary_topic"])
        .size()
        .reset_index(name="count")
        .sort_values(["venue", "year", "count", "primary_topic"], ascending=[True, True, False, True])
    )
    venue_year_counts["topic_label_en"] = venue_year_counts["primary_topic"].map(en_label_map)
    venue_year_counts["topic_label_zh"] = venue_year_counts["primary_topic"].map(zh_label_map)
    venue_year_counts["topic_label_bilingual"] = venue_year_counts["primary_topic"].map(bilingual_label_map)

    primary_counts.to_csv(freq_dir / "primary_topic_counts_by_venue.csv", index=False)
    all_counts.to_csv(freq_dir / "all_topic_counts_by_venue.csv", index=False)
    year_counts.to_csv(freq_dir / "primary_topic_counts_by_year.csv", index=False)
    venue_year_counts.to_csv(freq_dir / "primary_topic_counts_by_venue_year.csv", index=False)

    for venue, group in primary_counts.groupby("venue"):
        group.to_csv(freq_dir / f"{venue}_primary_topic_counts.csv", index=False)
    for venue, group in all_counts.groupby("venue"):
        group.to_csv(freq_dir / f"{venue}_all_topic_counts.csv", index=False)


def plot_figures(classified: pd.DataFrame, topics: list[TopicSpec], output_dir: Path, top_k: int) -> None:
    if classified.empty:
        return
    figures_dir = ensure_dir(output_dir / "figures")
    sns.set_theme(style="whitegrid")
    configure_plot_fonts()
    plot_label_map = topic_plot_label_map(topics)

    frame = classified.copy()
    overall_topic_order = frame["primary_topic"].value_counts().head(top_k).index.tolist()

    venue_pivot = (
        frame[frame["primary_topic"].isin(overall_topic_order)]
        .groupby(["venue", "primary_topic"])
        .size()
        .reset_index(name="count")
        .pivot(index="venue", columns="primary_topic", values="count")
        .fillna(0)
    )
    venue_pivot = venue_pivot.reindex(index=sorted(venue_pivot.index), columns=overall_topic_order)
    venue_pivot.index = [VENUE_DISPLAY_NAMES[item] for item in venue_pivot.index]
    venue_pivot.columns = [format_plot_label(plot_label_map[item]) for item in venue_pivot.columns]

    if len(venue_pivot.index) == 1:
        plot_data = venue_pivot.T
        fig, ax = plt.subplots(figsize=(5.6, max(7.5, len(plot_data.index) * 0.72)))
        sns.heatmap(
            plot_data,
            cmap="Reds",
            linewidths=0.4,
            annot=len(plot_data.index) <= 16,
            fmt=".0f",
            ax=ax,
            cbar_kws={"label": "Paper Count"},
        )
        ax.set_title("Primary Topic Frequency by Venue")
        ax.set_xlabel("Venue")
        ax.set_ylabel("Primary Topic")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0, labelsize=11)
    else:
        fig, ax = plt.subplots(
            figsize=(max(13, len(overall_topic_order) * 1.25), max(4.8, len(venue_pivot.index) * 1.05 + 2.2))
        )
        sns.heatmap(
            venue_pivot,
            cmap="Reds",
            linewidths=0.4,
            annot=len(overall_topic_order) <= 12 and len(venue_pivot.index) <= 8,
            fmt=".0f",
            ax=ax,
            cbar_kws={"label": "Paper Count"},
        )
        ax.set_title("Primary Topic Frequency by Venue")
        ax.set_xlabel("Primary Topic")
        ax.set_ylabel("Venue")
        ax.tick_params(axis="x", rotation=0, labelsize=10)
        ax.tick_params(axis="y", rotation=0)
    fig.savefig(figures_dir / "venue_topic_heatmap.png", dpi=220, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    year_pivot = (
        frame[frame["primary_topic"].isin(overall_topic_order)]
        .groupby(["year", "primary_topic"])
        .size()
        .reset_index(name="count")
        .pivot(index="year", columns="primary_topic", values="count")
        .fillna(0)
        .sort_index()
    )
    year_pivot = year_pivot[overall_topic_order]
    year_pivot.columns = [plot_label_map[item] for item in year_pivot.columns]

    ax = year_pivot.plot(kind="bar", stacked=True, figsize=(12.5, 6.4), colormap="tab20")
    ax.set_title("Primary Topic Trend by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Paper Count")
    ax.legend(title="Primary Topic", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title_fontsize=9)
    plt.savefig(figures_dir / "year_topic_stacked_bar.png", dpi=220, bbox_inches="tight", pad_inches=0.2)
    plt.close()

    for venue, venue_frame in frame.groupby("venue"):
        top_topics = venue_frame["primary_topic"].value_counts().head(top_k).index.tolist()
        pivot = (
            venue_frame[venue_frame["primary_topic"].isin(top_topics)]
            .groupby(["year", "primary_topic"])
            .size()
            .reset_index(name="count")
            .pivot(index="year", columns="primary_topic", values="count")
            .fillna(0)
            .sort_index()
        )
        if pivot.empty:
            continue
        pivot = pivot[top_topics]
        plot_data = pivot.T
        plot_data.index = [format_plot_label(plot_label_map[item]) for item in plot_data.index]
        fig, ax = plt.subplots(figsize=(6.4, max(7.5, len(top_topics) * 0.72)))
        sns.heatmap(
            plot_data,
            cmap="Reds",
            linewidths=0.4,
            annot=len(top_topics) <= 16,
            fmt=".0f",
            ax=ax,
            cbar_kws={"label": "Paper Count"},
        )
        ax.set_title(f"{VENUE_DISPLAY_NAMES[venue]} Topic Trend")
        ax.set_xlabel("Year")
        ax.set_ylabel("Primary Topic")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0, labelsize=11)
        fig.savefig(figures_dir / f"{venue}_topic_trend_heatmap.png", dpi=220, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    text_frame = df.fillna("").astype(str)
    header = "| " + " | ".join(text_frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(text_frame.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_frame.values.tolist()]
    return "\n".join([header, separator, *rows])


def write_summary_report(classified: pd.DataFrame, topics: list[TopicSpec], output_dir: Path) -> None:
    if classified.empty:
        return
    label_map = topic_bilingual_label_map(topics)
    frame = classified.copy()
    frame["topic_label"] = frame["primary_topic"].map(label_map)

    lines = [
        "# Literature Survey Summary",
        "",
        f"- Total papers: {len(frame)}",
        f"- Venues: {', '.join(VENUE_DISPLAY_NAMES[item] for item in sorted(frame['venue'].unique()))}",
        f"- Years: {', '.join(str(item) for item in sorted(frame['year'].unique(), reverse=True))}",
        "",
        "## Paper Counts by Venue and Year",
        "",
    ]

    venue_year = (
        frame.groupby(["venue", "year"])
        .size()
        .reset_index(name="paper_count")
        .sort_values(["venue", "year"], ascending=[True, False])
    )
    venue_year["venue"] = venue_year["venue"].map(VENUE_DISPLAY_NAMES)
    lines.append(df_to_markdown(venue_year))
    lines.extend(["", "## Top Primary Topics by Venue", ""])

    for venue, group in frame.groupby("venue"):
        lines.append(f"### {VENUE_DISPLAY_NAMES[venue]}")
        lines.append("")
        top_topics = (
            group["topic_label"]
            .value_counts()
            .head(10)
            .rename_axis("topic")
            .reset_index(name="count")
        )
        lines.append(df_to_markdown(top_topics))
        lines.append("")

    lines.extend(["## Top Primary Topics by Year", ""])
    for year, group in frame.groupby("year"):
        lines.append(f"### {year}")
        lines.append("")
        top_topics = (
            group["topic_label"]
            .value_counts()
            .head(10)
            .rename_axis("topic")
            .reset_index(name="count")
        )
        lines.append(df_to_markdown(top_topics))
        lines.append("")

    lines.extend(["## Top Paper Types", ""])
    paper_types = (
        frame["paper_type"].value_counts().rename_axis("paper_type").reset_index(name="count")
    )
    lines.append(df_to_markdown(paper_types))
    lines.append("")

    (output_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_run_metadata(output_dir: Path, args: argparse.Namespace, raw_count: int, classified_count: int) -> None:
    metadata = {
        "mode": args.mode,
        "venues": args.venues,
        "years": args.years,
        "max_papers_per_venue_year": args.max_papers_per_venue_year,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "classify_workers": args.classify_workers,
        "top_k_topics": args.top_k_topics,
        "raw_paper_count": raw_count,
        "classified_paper_count": classified_count,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return pd.read_csv(path)


def main() -> int:
    load_dotenv()
    args = parse_args()
    args.venues = normalize_venues(args.venues)
    args.years = sorted(set(args.years), reverse=True)

    topics = load_topics()
    system_prompt_path = get_system_prompt_path(args.system_prompt)
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt file does not exist: {system_prompt_path}")
    run_name = slugify_run(args.venues, args.years)
    output_dir = ensure_dir(OUTPUT_ROOT / run_name)

    raw_path = output_dir / "raw_papers.csv"
    classified_path = output_dir / "classified_papers.csv"
    partial_classified_path = output_dir / "classified_papers.partial.csv"
    progress_path = output_dir / "progress.json"

    raw_frame = pd.DataFrame()
    classified_frame = pd.DataFrame()

    if args.mode in {"all", "scrape"}:
        raw_frame = scrape_all(
            venues=args.venues,
            years=args.years,
            max_papers=args.max_papers_per_venue_year,
            workers=args.workers,
        )
        raw_frame.to_csv(raw_path, index=False)
        print(f"Wrote raw papers to {raw_path}")

    if args.mode in {"all", "classify"}:
        if raw_frame.empty:
            raw_frame = read_csv_if_exists(raw_path)
        classify_model = (
            args.classify_model
            or os.getenv("OPENAI_CLASSIFY_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-5.5"
        )
        classified_frame = classify_papers(
            papers=raw_frame,
            topics=topics,
            system_prompt_path=system_prompt_path,
            model=classify_model,
            reasoning_effort=args.reasoning_effort,
            batch_size=args.batch_size,
            classify_workers=args.classify_workers,
            partial_path=partial_classified_path,
            progress_path=progress_path,
        )
        atomic_write_csv(classified_path, classified_frame)
        print(f"Wrote classified papers to {classified_path}")

    if args.mode in {"all", "visualize"}:
        if classified_frame.empty:
            classified_frame = read_csv_if_exists(classified_path)
        write_frequency_tables(classified_frame, topics, output_dir)
        plot_figures(classified_frame, topics, output_dir, top_k=args.top_k_topics)
        write_summary_report(classified_frame, topics, output_dir)
        print(f"Wrote tables, figures, and summary report to {output_dir}")

    write_run_metadata(
        output_dir=output_dir,
        args=args,
        raw_count=len(raw_frame) if not raw_frame.empty else (len(read_csv_if_exists(raw_path)) if raw_path.exists() else 0),
        classified_count=(
            len(classified_frame)
            if not classified_frame.empty
            else (len(read_csv_if_exists(classified_path)) if classified_path.exists() else 0)
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
