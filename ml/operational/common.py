"""Shared types and bounded HTTP helpers for operational source adapters."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx


LIVE = "LIVE"
REFERENCE = "REFERENCE"
DEMO = "DEMO"
UNAVAILABLE = "UNAVAILABLE"


@dataclass
class ProviderResult:
    source: str
    status: str
    data: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def request_timeout() -> float:
    try:
        return max(1.0, min(float(os.getenv("ANVAYA_OPERATIONAL_TIMEOUT_SECONDS", "8")), 30.0))
    except ValueError:
        return 8.0


async def get_json(url: str, params: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
    async with httpx.AsyncClient(
        timeout=timeout or request_timeout(),
        follow_redirects=True,
        headers={"User-Agent": "ANVAYA-operational/1.0 (weather-data-client)"},
    ) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Provider returned a non-object JSON response")
    if payload.get("error"):
        raise ValueError(str(payload.get("reason") or payload["error"]))
    return payload


def as_finite_number(value: Any, *, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Missing numeric field: {label}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric field: {label}") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"Non-finite numeric field: {label}")
    return number
