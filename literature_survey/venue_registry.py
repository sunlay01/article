from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VenueSpec:
    key: str
    display_name: str
    adapter: str
    aliases: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)


def load_venue_specs(path: Path) -> dict[str, VenueSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs: dict[str, VenueSpec] = {}
    for item in payload["venues"]:
        spec = VenueSpec(
            key=item["key"],
            display_name=item["display_name"],
            adapter=item["adapter"],
            aliases=tuple(item.get("aliases", [])),
            config=item.get("config", {}),
        )
        specs[spec.key] = spec
    return specs
