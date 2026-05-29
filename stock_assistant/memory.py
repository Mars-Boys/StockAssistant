from __future__ import annotations

"""复盘记忆库与连续性跟踪模块。"""

from collections import Counter
from pathlib import Path
from typing import Any
import json

import pandas as pd

from stock_assistant.market import MarketSnapshot, market_breadth

MAINLINE_SECTOR_LIMIT = 5


def load_review_memory(path: str | Path) -> list[dict[str, Any]]:
    """读取本地复盘记忆库。"""
    file_path = Path(path)
    if not file_path.exists():
        return []

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(payload, list):
        return []

    entries = [item for item in payload if isinstance(item, dict)]
    return sorted(entries, key=lambda item: str(item.get("memory_date", "")), reverse=True)


def save_review_memory_entry(path: str | Path, entry: dict[str, Any]) -> str:
    """按日期写入或更新一条复盘记忆。"""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    entries = load_review_memory(file_path)
    memory_date = str(entry.get("memory_date", ""))
    action = "created"

    updated_entries: list[dict[str, Any]] = []
    replaced = False
    for item in entries:
        if str(item.get("memory_date", "")) == memory_date:
            updated_entries.append(entry)
            replaced = True
        else:
            updated_entries.append(item)

    if not replaced:
        updated_entries.append(entry)
    else:
        action = "updated"

    updated_entries = sorted(updated_entries, key=lambda item: str(item.get("memory_date", "")), reverse=True)
    file_path.write_text(json.dumps(updated_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return action


def delete_review_memory_entries(path: str | Path, memory_dates: list[str]) -> int:
    """按日期删除一条或多条复盘记忆，返回实际删除数量。"""
    file_path = Path(path)
    if not file_path.exists():
        return 0

    targets = {str(item).strip() for item in memory_dates if str(item).strip()}
    if not targets:
        return 0

    entries = load_review_memory(file_path)
    remaining = [item for item in entries if str(item.get("memory_date", "")) not in targets]
    removed_count = len(entries) - len(remaining)
    if removed_count == 0:
        return 0

    file_path.write_text(json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")
    return removed_count


def build_review_memory_entry(
    snapshot: MarketSnapshot,
    report: dict[str, Any],
    report_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把当日盘后分析压缩成适合长期保存的记忆条目。"""
    breadth = market_breadth(snapshot.stocks)
    sector_views = report.get("sector_views", []) if report else []
    compact_sector_views = [_compact_sector_view(item) for item in sector_views if isinstance(item, dict)]
    news_watch = report.get("news_watch", []) if report else []

    return {
        "memory_date": snapshot.fetched_at.strftime("%Y-%m-%d"),
        "fetched_at": snapshot.fetched_at.strftime("%Y-%m-%d %H:%M:%S"),
        "source": snapshot.source,
        "market_tone": str(report.get("market_tone", "待判断")),
        "summary": str(report.get("summary", "")),
        "key_points": _ensure_string_list(report.get("key_points")),
        "sector_views": compact_sector_views,
        "mainline_sectors": _mainline_sectors_from_snapshot(snapshot),
        "news_watch": [_compact_news_watch(item) for item in news_watch if isinstance(item, dict)],
        "next_focus": _ensure_string_list(report.get("next_focus")),
        "risk_flags": _ensure_string_list(report.get("risk_flags")),
        "breadth": breadth,
        "report_meta": {"model": (report_meta or {}).get("model", ""), "structured_mode": (report_meta or {}).get("structured_mode", "")},
    }


def review_memory_overview(entries: list[dict[str, Any]], current_entry: dict[str, Any] | None = None) -> dict[str, Any]:
    """汇总记忆库概况与连续性指标。"""
    recent_entries = entries[:10]
    recurring_sectors = _counter_items(_flatten_mainline_sectors(recent_entries), limit=5)
    recurring_risks = _counter_items(_flatten_strings(recent_entries, "risk_flags"), limit=5)
    recurring_focus = _counter_items(_flatten_strings(recent_entries, "next_focus"), limit=5)

    continuity = continuity_comparison(entries, current_entry=current_entry)
    dominant_sector = recurring_sectors[0]["label"] if recurring_sectors else "暂无"

    return {
        "saved_days": len(entries),
        "latest_date": entries[0].get("memory_date", "-") if entries else "-",
        "dominant_sector": dominant_sector,
        "recurring_sectors": recurring_sectors,
        "recurring_risks": recurring_risks,
        "recurring_focus": recurring_focus,
        "continuity": continuity,
    }


def continuity_comparison(entries: list[dict[str, Any]], current_entry: dict[str, Any] | None = None) -> dict[str, Any]:
    """比较当前复盘与上一期，输出连续性摘要。"""
    if current_entry is not None and entries:
        current_date = str(current_entry.get("memory_date", ""))
        previous = next(
            (entry for entry in entries if str(entry.get("memory_date", "")) != current_date),
            None,
        )
        current = current_entry
        if previous is not None:
            mode = "current_vs_latest_saved"
        else:
            return {
                "mode": "baseline_only",
                "current_date": current_date,
                "previous_date": "",
                "current_tone": str(current.get("market_tone", "")),
                "previous_tone": "",
                "tone_change": "基线已建立",
                "shared_sectors": [],
                "shared_risks": [],
                "new_sectors": _mainline_sector_names(current),
                "new_risks": sorted(_ensure_string_list(current.get("risk_flags"))),
            }
    elif len(entries) >= 2:
        current = entries[0]
        previous = entries[1]
        mode = "latest_saved_vs_previous_saved"
    elif len(entries) == 1:
        current = entries[0]
        return {
            "mode": "baseline_only",
            "current_date": str(current.get("memory_date", "")),
            "previous_date": "",
            "current_tone": str(current.get("market_tone", "")),
            "previous_tone": "",
            "tone_change": "基线已建立",
            "shared_sectors": [],
            "shared_risks": [],
            "new_sectors": _mainline_sector_names(current),
            "new_risks": sorted(_ensure_string_list(current.get("risk_flags"))),
        }
    else:
        return {
            "mode": "insufficient_history",
            "current_date": (current_entry or {}).get("memory_date", ""),
            "previous_date": "",
            "shared_sectors": [],
            "shared_risks": [],
            "new_sectors": [],
            "new_risks": [],
            "tone_change": "",
        }

    current_sectors = set(_mainline_sector_names(current))
    previous_sectors = set(_mainline_sector_names(previous))
    current_risks = set(_ensure_string_list(current.get("risk_flags")))
    previous_risks = set(_ensure_string_list(previous.get("risk_flags")))

    return {
        "mode": mode,
        "current_date": str(current.get("memory_date", "")),
        "previous_date": str(previous.get("memory_date", "")),
        "current_tone": str(current.get("market_tone", "")),
        "previous_tone": str(previous.get("market_tone", "")),
        "tone_change": _tone_change(current.get("market_tone", ""), previous.get("market_tone", "")),
        "shared_sectors": sorted(item for item in current_sectors & previous_sectors if item),
        "new_sectors": sorted(item for item in current_sectors - previous_sectors if item),
        "shared_risks": sorted(item for item in current_risks & previous_risks if item),
        "new_risks": sorted(item for item in current_risks - previous_risks if item),
    }


def memory_timeline_frame(entries: list[dict[str, Any]], limit: int = 12) -> pd.DataFrame:
    """把记忆库转换为时间线表格。"""
    rows: list[dict[str, Any]] = []
    for entry in entries[:limit]:
        rows.append(
            {
                "日期": entry.get("memory_date", ""),
                "市场风格": entry.get("market_tone", ""),
                "主线板块": "、".join(_mainline_sector_names(entry)[:3]),
                "观察重点": "；".join(_ensure_string_list(entry.get("next_focus"))[:2]),
                "风险条数": len(_ensure_string_list(entry.get("risk_flags"))),
            }
        )
    return pd.DataFrame(rows)


def memory_detail_frame(entries: list[dict[str, Any]], limit: int = 12) -> pd.DataFrame:
    """输出更详细的记忆库视图。"""
    rows: list[dict[str, Any]] = []
    for entry in entries[:limit]:
        rows.append(
            {
                "日期": entry.get("memory_date", ""),
                "风格": entry.get("market_tone", ""),
                "摘要": entry.get("summary", ""),
                "重点板块": "、".join(_mainline_sector_names(entry)[:3]),
                "重点风险": "；".join(_ensure_string_list(entry.get("risk_flags"))[:2]),
            }
        )
    return pd.DataFrame(rows)


def _compact_sector_view(item: dict[str, Any]) -> dict[str, str]:
    return {
        "sector": str(item.get("sector", "")),
        "view": str(item.get("view", "")),
        "driver": str(item.get("driver", "")),
        "risk": str(item.get("risk", "")),
    }


def _mainline_sectors_from_snapshot(snapshot: MarketSnapshot, limit: int = MAINLINE_SECTOR_LIMIT) -> list[str]:
    """用当日领涨板块作为主线候选，避免把领跌板块误计入连续性。"""
    sectors = snapshot.sectors
    if sectors.empty or "name" not in sectors or "change_pct" not in sectors:
        return []

    scoped = sectors.copy()
    scoped["change_pct"] = pd.to_numeric(scoped["change_pct"], errors="coerce")
    leaders = scoped[scoped["change_pct"] > 0].sort_values("change_pct", ascending=False)
    return [str(name) for name in leaders["name"].head(limit).tolist() if str(name).strip()]


def _mainline_sector_names(entry: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for name in _ensure_string_list(entry.get("mainline_sectors")):
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _flatten_mainline_sectors(entries: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for entry in entries:
        values.extend(_mainline_sector_names(entry))
    return values


def _compact_news_watch(item: dict[str, Any]) -> dict[str, str]:
    return {
        "title": str(item.get("title", "")),
        "impact": str(item.get("impact", "")),
        "sentiment": str(item.get("sentiment", "")),
    }


def _ensure_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _flatten_strings(entries: list[dict[str, Any]], field_key: str) -> list[str]:
    values: list[str] = []
    for entry in entries:
        values.extend(_ensure_string_list(entry.get(field_key)))
    return values


def _counter_items(values: list[str], limit: int) -> list[dict[str, Any]]:
    counter = Counter(item for item in values if item)
    return [{"label": label, "count": count} for label, count in counter.most_common(limit)]


def _tone_change(current_tone: Any, previous_tone: Any) -> str:
    current = str(current_tone).strip()
    previous = str(previous_tone).strip()
    if not current or not previous:
        return ""
    if current == previous:
        return "市场风格延续"
    return f"{previous} -> {current}"
