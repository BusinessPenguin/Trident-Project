"""HTTP helpers with bounded retries for vendor ingestion."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional, Tuple

import requests


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _retry_after_seconds(resp: requests.Response) -> Optional[float]:
    val = resp.headers.get("Retry-After")
    if not val:
        return None
    try:
        return max(0.0, float(val))
    except Exception:
        return None


def _deterministic_jitter(seed: str, attempt: int, max_jitter: float = 0.25) -> float:
    key = f"{seed}:{attempt}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return float(max_jitter) * bucket


def request_json_with_retries(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
    max_attempts: int = 4,
    retry_budget_seconds: float = 20.0,
    backoff_base_seconds: float = 0.5,
    backoff_cap_seconds: float = 4.0,
    seed: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Execute GET request and return parsed JSON + retry metadata.
    Retries are capped and deterministic for reproducible behavior.
    """
    req_seed = seed or url
    started = time.monotonic()
    attempts = 0
    retries = 0
    last_error: Optional[Exception] = None

    while attempts < int(max_attempts):
        attempts += 1
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            status = int(resp.status_code)
            if status in RETRYABLE_STATUS:
                if attempts >= int(max_attempts):
                    resp.raise_for_status()
                retries += 1
                retry_after = _retry_after_seconds(resp)
                if retry_after is None:
                    backoff = min(
                        float(backoff_cap_seconds),
                        float(backoff_base_seconds) * (2 ** (attempts - 1)),
                    )
                    retry_after = backoff + _deterministic_jitter(req_seed, attempts)
                if (time.monotonic() - started + retry_after) > float(retry_budget_seconds):
                    resp.raise_for_status()
                time.sleep(max(0.0, retry_after))
                continue
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError("HTTP JSON payload is not an object")
            return data, {
                "attempts": attempts,
                "retries": retries,
                "status_code": status,
            }
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            if attempts >= int(max_attempts):
                break
            retries += 1
            sleep_s = min(
                float(backoff_cap_seconds),
                float(backoff_base_seconds) * (2 ** (attempts - 1)),
            ) + _deterministic_jitter(req_seed, attempts)
            if (time.monotonic() - started + sleep_s) > float(retry_budget_seconds):
                break
            time.sleep(max(0.0, sleep_s))
            continue
        except requests.RequestException as exc:
            last_error = exc
            break
        except ValueError as exc:
            last_error = exc
            break

    if last_error is None:
        raise RuntimeError("HTTP request failed without explicit exception")
    raise last_error

