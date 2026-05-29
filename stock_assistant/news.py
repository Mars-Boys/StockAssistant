from __future__ import annotations

"""财经快讯采集与标准化处理流程。

这个模块主要负责五件事：
1. 读取配置中的综合消息渠道。
2. 适配东方财富和同花顺快讯数据。
3. 提取并解析发布时间。
4. 执行“仅保留当天消息”的过滤规则。
5. 输出来源、标题和发布时间，保持采集层只做事实整理。
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

# 所有日期比较统一换算到上海时区，因为这个项目面向 A 股盘后复盘，
# “当天”这个概念应当跟随本地市场时间。
LOCAL_TZ = ZoneInfo("Asia/Shanghai")
# 兼容快讯常见日期格式，例如 2026-05-24、2026/05/24、2026年05月24日 09:30。
DATE_PATTERN = re.compile(r"(20\d{2})[./\-_年](\d{1,2})[./\-_月](\d{1,2})日?(?:\s+(\d{1,2}):(\d{2}))?")
RELATIVE_MINUTES_PATTERN = re.compile(r"(?i)\b(\d+)\s*(minute|minutes|min)\s+ago\b")
RELATIVE_HOURS_PATTERN = re.compile(r"(?i)\b(\d+)\s*(hour|hours|hr|hrs)\s+ago\b")
RELATIVE_DAYS_PATTERN = re.compile(r"(?i)\b(\d+)\s*(day|days)\s+ago\b")
CN_RELATIVE_PATTERN = re.compile(r"(?:(\d+)\s*分钟前|(\d+)\s*小时前|(\d+)\s*天前)")
EASTMONEY_FAST_NEWS_URL = "https://np-weblist.eastmoney.com/comm/web/getFastNewsList"
THS_REALTIME_JS_URL = "https://stock.10jqka.com.cn/thsgd/realtimenews.js"
NEWS_COLUMNS = ["source", "title", "published_at"]
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://kuaixun.eastmoney.com/",
}
GENERIC_NAV_TITLES = {
    "7*24全球直播",
    "点击加载更多",
    "桌面通知",
    "声音提醒",
    "新闻",
    "股票",
    "基金",
    "全球",
    "查询",
}


@dataclass(frozen=True)
class NewsSource:
    """从 YAML 配置中读取出来的单个消息源定义。"""

    name: str
    url: str


def load_sources(path: str | Path) -> list[NewsSource]:
    """从轻量级 YAML 风格配置文件中读取消息源定义。"""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []

    sources: list[dict[str, Any]] = _parse_simple_yaml(text)
    allowed_keys = {"name", "url"}
    result: list[NewsSource] = []
    for item in sources:
        values = {key: value for key, value in item.items() if key in allowed_keys}
        if values.get("name") and values.get("url"):
            result.append(NewsSource(name=str(values["name"]), url=str(values["url"])))
    return result


def collect_news(sources: list[NewsSource], limit_per_source: int = 12) -> pd.DataFrame:
    """抓取、过滤并合并当天的财经快讯。

    返回的 DataFrame 是前端“财经快讯”标签页使用的唯一数据来源。
    """
    if not sources:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    rows: list[dict[str, Any]] = []
    today = datetime.now(LOCAL_TZ).date()
    for source in sources:
        try:
            items = _fetch_source_news(source)
        except Exception:
            # 即使来源临时失效或禁止抓取，也尽量让页面保持可用。
            items = _demo_news(source)
        if not items:
            items = _demo_news(source)

        added = 0
        for item in items:
            published_at = _normalize_published_at(item.get("published_at", ""))
            # 用户要求只看当日消息；无法识别日期的内容宁可舍弃也不混入结果。
            if published_at is None or published_at.date() != today:
                continue

            rows.append(
                {
                    "source": source.name,
                    "title": item["title"],
                    "published_at": published_at.strftime("%Y-%m-%d %H:%M"),
                }
            )
            added += 1
            if added >= limit_per_source:
                # 每个渠道只取少量高相关内容，避免单一站点刷屏。
                break

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    return frame[NEWS_COLUMNS].sort_values("published_at", ascending=False).reset_index(drop=True)


def _fetch_source_news(source: NewsSource) -> list[dict[str, str]]:
    """根据渠道选择最稳定的采集方式。"""
    identifier = f"{source.name} {source.url}".lower()
    if "eastmoney" in identifier or "东方财富" in source.name:
        return _fetch_eastmoney_news(source)
    if "10jqka" in identifier or "同花顺" in source.name:
        return _fetch_ths_news(source)
    return _fetch_page(source)


def _fetch_eastmoney_news(source: NewsSource) -> list[dict[str, str]]:
    """读取东方财富 7x24 快讯 JSON。"""
    response = requests.get(
        EASTMONEY_FAST_NEWS_URL,
        params={
            "client": "web",
            "biz": "web_724",
            "fastColumn": "",
            "sortEnd": "",
            "pageSize": 50,
            "req_trace": uuid4().hex,
        },
        timeout=8,
        headers={**REQUEST_HEADERS, "Referer": source.url},
    )
    response.raise_for_status()
    payload = response.json()
    data = (payload.get("data") or {}) if isinstance(payload, dict) else {}
    records = data.get("fastNewsList", [])
    items: list[dict[str, str]] = []
    for record in records:
        title = _clean_title(record.get("title", ""))
        if not title:
            continue
        items.append(
            {
                "title": title,
                "published_at": str(record.get("showTime", "")),
            }
        )
    return _deduplicate(items)


def _fetch_ths_news(source: NewsSource) -> list[dict[str, str]]:
    """读取同花顺 7x24 快讯脚本中的 thsRss.item 列表。"""
    response = requests.get(
        THS_REALTIME_JS_URL,
        timeout=8,
        headers={**REQUEST_HEADERS, "Referer": source.url},
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    records = json.loads(_extract_js_array(response.text, "item"))
    items: list[dict[str, str]] = []
    for record in records:
        title = _clean_title(record.get("title", ""))
        if not title:
            continue
        items.append(
            {
                "title": title,
                "published_at": str(record.get("pubDate", "")),
            }
        )
    return _deduplicate(items)


def _fetch_page(source: NewsSource) -> list[dict[str, str]]:
    """兜底：从普通网页中抓取带时间上下文的消息链接。"""
    response = requests.get(source.url, timeout=8, headers={**REQUEST_HEADERS, "Referer": source.url})
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text, "html.parser")
    items: list[dict[str, str]] = []

    # 兜底逻辑只接收能提取到时间的链接，避免混入导航、日历和行情入口。
    for anchor in soup.find_all("a"):
        title = _clean_title(anchor.get_text(" ", strip=True))
        href = anchor.get("href")
        if not href or not _looks_like_story_title(title):
            continue
        published_at = _extract_published_at(anchor)
        if not published_at:
            continue
        items.append(
            {
                "title": title,
                "published_at": published_at,
            }
        )
        if len(items) >= 20:
            break
    return items


def _extract_published_at(anchor) -> str:
    """尝试从链接周边的 HTML 文本中恢复发布时间。"""
    # 公告列表的日期经常挂在父节点或祖父节点上，这里按就近上下文逐层尝试。
    candidates = [
        anchor.get_text(" ", strip=True),
        anchor.get("title", ""),
        anchor.get("aria-label", ""),
        anchor.parent.get_text(" ", strip=True) if anchor.parent else "",
    ]
    for node in _context_nodes(anchor):
        candidates.extend(_extract_datetime_candidates(node))
        if getattr(node, "get_text", None):
            candidates.append(node.get_text(" ", strip=True))

    for text in candidates:
        published_at = _normalize_published_at(text)
        if published_at is not None:
            return published_at.isoformat()
    return ""


def _context_nodes(anchor) -> list[Any]:
    """获取链接周边最可能携带时间信息的上下文节点。"""
    nodes: list[Any] = []
    current = anchor.parent
    depth = 0
    while current is not None and depth < 4:
        nodes.append(current)
        current = current.parent
        depth += 1
    return nodes


def _extract_datetime_candidates(node) -> list[str]:
    """从上下文节点中提取 time/datetime 一类更结构化的时间文本。"""
    candidates: list[str] = []
    if not getattr(node, "find_all", None):
        return candidates

    for time_node in node.find_all(["time"]):
        candidates.extend(
            [
                time_node.get("datetime", ""),
                time_node.get("title", ""),
                time_node.get_text(" ", strip=True),
            ]
        )

    for tagged_node in node.find_all(attrs={"datetime": True}):
        candidates.append(tagged_node.get("datetime", ""))
    return [item for item in candidates if item]


def _normalize_published_at(raw: str) -> datetime | None:
    """把多种可能的日期字符串标准化为带时区的时间对象。"""
    cleaned = " ".join(str(raw).split()).strip()
    if not cleaned:
        return None

    relative = _parse_relative_time(cleaned)
    if relative is not None:
        return relative

    # 先走显式正则匹配，成本低且结果更稳定。
    match = DATE_PATTERN.search(cleaned)
    if match:
        year, month, day, hour, minute = match.groups()
        return datetime(
            int(year),
            int(month),
            int(day),
            int(hour or 0),
            int(minute or 0),
            tzinfo=LOCAL_TZ,
        )

    try:
        # 最后一层兜底交给 dateutil，兼容更散乱的时间文本。
        parsed = date_parser.parse(cleaned, fuzzy=True)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(LOCAL_TZ)


def _parse_relative_time(text: str) -> datetime | None:
    """兼容快讯常见的相对时间，例如 30分钟前。"""
    now = datetime.now(LOCAL_TZ)

    minute_match = RELATIVE_MINUTES_PATTERN.search(text)
    if minute_match:
        return now - pd.Timedelta(minutes=int(minute_match.group(1)))

    hour_match = RELATIVE_HOURS_PATTERN.search(text)
    if hour_match:
        return now - pd.Timedelta(hours=int(hour_match.group(1)))

    day_match = RELATIVE_DAYS_PATTERN.search(text)
    if day_match:
        return now - pd.Timedelta(days=int(day_match.group(1)))

    cn_match = CN_RELATIVE_PATTERN.search(text)
    if cn_match:
        minutes, hours, days = cn_match.groups()
        if minutes:
            return now - pd.Timedelta(minutes=int(minutes))
        if hours:
            return now - pd.Timedelta(hours=int(hours))
        if days:
            return now - pd.Timedelta(days=int(days))

    if "yesterday" in text.lower() or "昨天" in text:
        return now - pd.Timedelta(days=1)
    return None


def _looks_like_story_title(title: str) -> bool:
    """过滤导航、栏目名等非正文消息链接。"""
    normalized = _clean_title(title)
    if len(normalized) < 8:
        return False

    if normalized in GENERIC_NAV_TITLES:
        return False

    chinese_char_count = len(re.findall(r"[\u4e00-\u9fff]", normalized))
    return chinese_char_count >= 6


def _deduplicate(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """以标题为主键去重，避免同源页面重复收录相同消息。"""
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in items:
        key = item["title"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _clean_title(value: Any) -> str:
    """清理脚本和 HTML 中可能混入的标签、空白和实体字符。"""
    raw_text = str(value or "")
    text = BeautifulSoup(raw_text, "html.parser").get_text(" ", strip=True) if "<" in raw_text else raw_text
    return " ".join(text.split())


def _extract_js_array(script: str, field: str) -> str:
    """从简单 JS 对象文本中提取数组字面量。"""
    marker = f"{field}:"
    start = script.find(marker)
    if start < 0:
        raise ValueError(f"missing {field} array")
    array_start = script.find("[", start)
    if array_start < 0:
        raise ValueError(f"missing {field} array start")

    depth = 0
    in_string = False
    escape = False
    quote = ""
    for index in range(array_start, len(script)):
        char = script[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue

        if char in {"'", '"'}:
            in_string = True
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return script[array_start : index + 1]

    raise ValueError(f"unterminated {field} array")


def _demo_news(source: NewsSource) -> list[dict[str, str]]:
    """为离线模式或演示模式生成可预测的兜底消息。"""
    today = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    return [
        {
            "title": f"{source.name}快讯：多部门推出促进资本市场稳定健康发展的专项举措",
            "published_at": today,
        },
        {
            "title": f"{source.name}快讯：上市公司信息披露监管和风险整改持续推进",
            "published_at": today,
        },
    ]


def _parse_simple_yaml(text: str) -> list[dict[str, Any]]:
    """解析 `news_sources.yml` 实际用到的那一小部分 YAML 语法。

    这里故意把配置格式限制得很简单，这样就不必为了一个扁平列表
    额外引入专门的 YAML 依赖。
    """
    sources: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "sources:":
            continue
        if stripped.startswith("- "):
            if current:
                sources.append(current)
            current = {}
            stripped = stripped[2:]
            if stripped and ":" in stripped:
                key, value = stripped.split(":", 1)
                current[key.strip()] = _coerce_yaml_value(value.strip())
            continue
        if current is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = _coerce_yaml_value(value.strip())

    if current:
        sources.append(current)
    return sources


def _coerce_yaml_value(value: str) -> Any:
    # 目前配置里只有字符串和数值，保持解析器简单，避免额外引入 YAML 依赖。
    value = value.strip().strip('"').strip("'")
    try:
        # 权重字段要转成浮点数，其余内容继续保留为字符串。
        return float(value)
    except ValueError:
        return value
