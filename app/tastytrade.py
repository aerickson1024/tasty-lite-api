"""Tastytrade market data client: OAuth token refresh, quotes and IV rank."""

import asyncio
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

BASE_URL = "https://api.tastyworks.com"
USER_AGENT = "tasty-lite-api/1.0"
DEFAULT_SYMBOLS = "SPY,QQQ,AAPL,TSLA,NVDA"
DEFAULT_TOKEN_LIFETIME_S = 900
TOKEN_REFRESH_MARGIN_S = 60
TIMEOUT_S = 10.0


class TastytradeError(Exception):
    """An upstream request failed."""


class TastytradeNotConfigured(TastytradeError):
    """Credentials are missing from the environment."""


async def _refresh_access_token(
    http: httpx.AsyncClient, secret: str, refresh: str
) -> tuple[str, float]:
    resp = await http.post(
        f"{BASE_URL}/oauth/token",
        headers={"User-Agent": USER_AGENT},
        json={"grant_type": "refresh_token", "refresh_token": refresh, "client_secret": secret},
    )
    if resp.status_code != 200:
        raise TastytradeError(f"token refresh failed with HTTP {resp.status_code}")
    body = resp.json()
    return body["access_token"], float(body.get("expires_in", DEFAULT_TOKEN_LIFETIME_S))


class _TokenCache:
    def __init__(self) -> None:
        self._token = ""
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    async def get(self, http: httpx.AsyncClient, secret: str, refresh: str) -> str:
        async with self._lock:
            if time.monotonic() >= self._expires_at:
                self._token, lifetime = await _refresh_access_token(http, secret, refresh)
                self._expires_at = time.monotonic() + lifetime - TOKEN_REFRESH_MARGIN_S
            return self._token


_tokens = _TokenCache()


def _settings() -> tuple[str, str, list[str]]:
    secret = os.environ.get("TASTY_CLIENT_SECRET", "")
    refresh = os.environ.get("TASTY_REFRESH_TOKEN", "")
    if not secret or not refresh:
        raise TastytradeNotConfigured("set TASTY_CLIENT_SECRET and TASTY_REFRESH_TOKEN")
    raw = os.environ.get("TASTY_SYMBOLS", DEFAULT_SYMBOLS)
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return secret, refresh, symbols


def _num(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def merge_stocks(symbols: list[str], quotes: list[dict], metrics: list[dict]) -> list[dict]:
    quote_by_symbol = {q["symbol"]: q for q in quotes}
    metric_by_symbol = {m["symbol"]: m for m in metrics}
    rows = []
    for symbol in symbols:
        quote = quote_by_symbol.get(symbol, {})
        metric = metric_by_symbol.get(symbol, {})
        rows.append(
            {
                "symbol": symbol,
                "last": _num(quote.get("last")),
                "volume": _num(quote.get("volume")),
                "iv_rank": _num(metric.get("implied-volatility-index-rank")),
            }
        )
    return rows


def _items(resp: httpx.Response, name: str) -> list[dict]:
    if resp.status_code != 200:
        raise TastytradeError(f"{name} request failed with HTTP {resp.status_code}")
    return resp.json()["data"]["items"]


async def fetch_stocks() -> list[dict]:
    secret, refresh, symbols = _settings()
    joined = ",".join(symbols)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as http:
            token = await _tokens.get(http, secret, refresh)
            headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
            quotes, metrics = await asyncio.gather(
                http.get(f"{BASE_URL}/market-data/by-type", params={"equity": joined}, headers=headers),
                http.get(f"{BASE_URL}/market-metrics", params={"symbols": joined}, headers=headers),
            )
    except httpx.HTTPError as exc:
        raise TastytradeError(f"network error: {exc}") from exc
    return merge_stocks(symbols, _items(quotes, "market-data"), _items(metrics, "market-metrics"))
