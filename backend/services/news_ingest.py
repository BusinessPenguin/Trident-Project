"""News ingestion helpers for Project Trident."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.services.news_api import NewsArticle, classify_macro_tag


def _to_utc(ts: Optional[datetime]) -> Optional[datetime]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _canon_text(val: Optional[str]) -> str:
    return " ".join((val or "").strip().lower().split())


def _summary_from_ai_meta(ai_meta_raw: Optional[str]) -> str:
    if not ai_meta_raw:
        return ""
    try:
        parsed = json.loads(ai_meta_raw)
    except Exception:
        return ""
    if not isinstance(parsed, dict):
        return ""
    return _canon_text(parsed.get("summary"))


def _should_update_existing(
    existing: Dict[str, object],
    incoming_ts: Optional[datetime],
    incoming_title: Optional[str],
    incoming_ai_meta_json: Optional[str],
) -> bool:
    old_ts = _to_utc(existing.get("published_at"))  # type: ignore[arg-type]
    new_ts = _to_utc(incoming_ts)
    if old_ts is not None and new_ts is not None and new_ts > old_ts:
        return True
    old_title = _canon_text(existing.get("title"))  # type: ignore[arg-type]
    new_title = _canon_text(incoming_title)
    if new_title and new_title != old_title:
        return True
    old_summary = _summary_from_ai_meta(existing.get("ai_meta"))  # type: ignore[arg-type]
    new_summary = _summary_from_ai_meta(incoming_ai_meta_json)
    if new_summary and new_summary != old_summary:
        return True
    return False


def ingest_news(
    con,
    symbol: str | None,
    articles: List[NewsArticle],
    lane: str = "symbol",
    stats: Optional[dict] = None,
) -> int:
    """
    Insert news articles for a symbol into DuckDB, ignoring duplicates by URL.
    """
    count = 0
    existing_by_url: Dict[str, Dict[str, object]] = {}
    duplicate_skips = 0
    updates = 0
    inserts = 0
    macro_untagged_skips = 0
    url_candidates = [a.url for a in articles if getattr(a, "url", None)]
    if url_candidates:
        chunk_size = 500
        for i in range(0, len(url_candidates), chunk_size):
            chunk = url_candidates[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            rows = con.execute(
                f"SELECT url, published_at, title, ai_meta FROM news_items WHERE url IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                if row and row[0]:
                    existing_by_url[row[0]] = {
                        "published_at": row[1],
                        "title": row[2],
                        "ai_meta": row[3],
                    }

    for art in articles:
        if not art.url:
            continue
        if lane == "macro" and not art.macro_tag:
            summary = ""
            if art.ai_meta:
                summary = art.ai_meta.get("summary") or ""
            art.macro_tag = classify_macro_tag(art.title or "", summary)
            if not art.macro_tag:
                # Skip untagged macro rows to keep lane clean.
                macro_untagged_skips += 1
                continue
        ai_meta_json = json.dumps(art.ai_meta) if art.ai_meta else None
        existing = existing_by_url.get(art.url)
        if existing is not None:
            if not _should_update_existing(
                existing=existing,
                incoming_ts=art.ts,
                incoming_title=art.title,
                incoming_ai_meta_json=ai_meta_json,
            ):
                duplicate_skips += 1
                continue
            try:
                con.execute(
                    """
                    UPDATE news_items
                    SET symbol = ?,
                        published_at = ?,
                        source = ?,
                        title = ?,
                        polarity = ?,
                        ai_meta = ?,
                        lane = ?,
                        macro_tag = ?
                    WHERE url = ?
                    """,
                    [
                        symbol,
                        art.ts,
                        art.source,
                        art.title,
                        art.polarity,
                        ai_meta_json,
                        lane,
                        art.macro_tag,
                        art.url,
                    ],
                )
            except Exception:
                con.execute(
                    """
                    UPDATE news_items
                    SET symbol = ?,
                        published_at = ?,
                        source = ?,
                        title = ?,
                        polarity = ?,
                        ai_meta = ?
                    WHERE url = ?
                    """,
                    [
                        symbol,
                        art.ts,
                        art.source,
                        art.title,
                        art.polarity,
                        ai_meta_json,
                        art.url,
                    ],
                )
            existing_by_url[art.url] = {
                "published_at": art.ts,
                "title": art.title,
                "ai_meta": ai_meta_json,
            }
            updates += 1
            count += 1
            continue
        try:
            con.execute(
                """
                INSERT OR IGNORE INTO news_items
                    (symbol, published_at, source, url, title, polarity, ai_meta, lane, macro_tag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    symbol,
                    art.ts,
                    art.source,
                    art.url,
                    art.title,
                    art.polarity,
                    ai_meta_json,
                    lane,
                    art.macro_tag,
                ],
            )
        except Exception:
            # Fallback for legacy schemas without lane/macro_tag columns.
            con.execute(
                """
                INSERT OR IGNORE INTO news_items
                    (symbol, published_at, source, url, title, polarity, ai_meta)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    symbol,
                    art.ts,
                    art.source,
                    art.url,
                    art.title,
                    art.polarity,
                    ai_meta_json,
                ],
            )
        existing_by_url[art.url] = {
            "published_at": art.ts,
            "title": art.title,
            "ai_meta": ai_meta_json,
        }
        inserts += 1
        count += 1
    if stats is not None:
        stats.clear()
        stats.update(
            {
                "input_articles": int(len(articles)),
                "inserted": int(inserts),
                "updated": int(updates),
                "dupe_skips": int(duplicate_skips),
                "macro_untagged_skips": int(macro_untagged_skips),
                "written_total": int(count),
            }
        )
    return count
