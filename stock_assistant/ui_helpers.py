from __future__ import annotations

"""Streamlit 页面层的轻量辅助函数。

这些函数不负责拉取行情、采集新闻或调用 AI，只处理展示层常见的格式化、
图表选点解析和本地环境读取。
"""

import hashlib
import os
from typing import Any

import pandas as pd
import streamlit as st

from stock_assistant.ai_insights import context_preview


def format_pct(value: float | int | None) -> str:
    """将百分比数值格式化为表格展示文本。"""
    if pd.isna(value):
        return "-"
    return f"{value:.2f}%"


def format_number(value: float | int | None, digits: int = 2) -> str:
    """将可空数值格式化为固定小数位文本。"""
    if pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def metric_delta(value: float | int | None) -> str:
    """格式化指标卡的涨跌文本，空值时返回空字符串。"""
    return "" if pd.isna(value) else f"{value:.2f}%"


def existing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """仅保留 DataFrame 中真实存在的列。"""
    return [column for column in columns if column in frame.columns]


def selected_sector_name(event: Any) -> str | None:
    """从 Plotly 选点事件中提取板块名称。"""
    if not event or not event.selection.points:
        return None
    point = event.selection.points[-1]
    customdata = point.get("customdata") or []
    if customdata and customdata[0]:
        return str(customdata[0])
    for key in ("label", "id"):
        value = point.get(key)
        if value:
            return str(value)
    return None


def sector_selection_token(event: Any, chart_nonce: int) -> str | None:
    """把当前选点和图表实例一起编码，稳定识别每次新点击。"""
    if not event or not event.selection.points:
        return None
    point = event.selection.points[-1]
    label = selected_sector_name(event) or ""
    point_number = point.get("point_number", point.get("point_index", ""))
    return f"{chart_nonce}:{label}:{point_number}"


def trim_sector_history(frame: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """按自然日窗口裁剪趋势数据，数据不足时保留至少最近两条。"""
    if frame.empty or "date" not in frame:
        return frame
    scoped = frame.sort_values("date").copy()
    cutoff = scoped["date"].max() - pd.Timedelta(days=max(window_days - 1, 1))
    scoped = scoped[scoped["date"] >= cutoff].copy()
    if len(scoped) >= 2:
        return scoped.reset_index(drop=True)
    return frame.sort_values("date").tail(min(len(frame), 2)).reset_index(drop=True)


def default_deepseek_api_key() -> str:
    """从系统环境变量中读取 DeepSeek Key。"""
    return os.getenv("DEEPSEEK_API_KEY", "")


def ai_context_signature(context: dict[str, Any]) -> str:
    """为当前 AI 分析上下文计算一个稳定签名。"""
    payload = context_preview(context)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def render_report_section(title: str, items: list[str], empty_text: str) -> None:
    """统一渲染 AI 输出中的简单列表段落。"""
    st.markdown(f"**{title}**")
    if not items:
        st.caption(empty_text)
        return
    for item in items:
        st.markdown(f"- {item}")


def format_structured_mode(value: Any) -> str:
    """把内部结构化调用模式转换为适合页面展示的短文本。"""
    mode = str(value or "").strip()
    if not mode:
        return ""
    if mode == "json_output_thinking":
        return "JSON Output（思考模式）"
    if mode == "strict_function_call":
        return "Strict Function Calling"
    if mode.startswith("json_fallback"):
        return "JSON Output fallback"
    if mode == "plain_text_fallback":
        return "普通文本 fallback"
    return mode
