from __future__ import annotations

"""盘后 AI 分析与 DeepSeek 接入模块。"""

from dataclasses import dataclass
import json
from typing import Any

import pandas as pd
import requests

from stock_assistant.market import MarketSnapshot, market_breadth, top_rows
from stock_assistant.prompts.market_review import (
    QNA_SCHEMA,
    QNA_SYSTEM_PROMPT,
    REPORT_SCHEMA,
    REPORT_SYSTEM_PROMPT,
    VALID_NEWS_SENTIMENTS,
)


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_BETA_BASE_URL = "https://api.deepseek.com/beta"


class AIServiceError(RuntimeError):
    """统一封装 AI 服务调用与解析阶段的异常。"""


@dataclass(frozen=True)
class DeepSeekConfig:
    """DeepSeek 接口所需的最小配置。"""

    api_key: str
    model: str = "deepseek-v4-pro"
    thinking_enabled: bool = True
    reasoning_effort: str = "high"
    temperature: float = 0.2
    max_tokens: int = 1800
    timeout: int = 90


def build_analysis_context(
    snapshot: MarketSnapshot,
    news: pd.DataFrame,
    sector_limit: int = 8,
    stock_limit: int = 8,
    news_limit: int = 10,
) -> dict[str, Any]:
    """把页面已有结构压缩成更适合大模型消费的上下文。"""
    breadth = market_breadth(snapshot.stocks)
    top_sectors = top_rows(snapshot.sectors, "change_pct", sector_limit)
    weak_sectors = top_rows(snapshot.sectors, "change_pct", sector_limit, ascending=True)
    strong_stocks = top_rows(snapshot.stocks, "change_pct", stock_limit)
    weak_stocks = top_rows(snapshot.stocks, "change_pct", stock_limit, ascending=True)

    return {
        "snapshot": {
            "fetched_at": snapshot.fetched_at.strftime("%Y-%m-%d %H:%M:%S"),
            "source": snapshot.source,
            "error": snapshot.error,
        },
        "market_breadth": breadth,
        "indices": _frame_records(
            snapshot.indices,
            ["name", "price", "change_pct", "turnover"],
            limit=10,
        ),
        "top_sectors": _frame_records(
            top_sectors,
            ["name", "change_pct", "up_count", "down_count", "leader", "leader_change_pct"],
            limit=sector_limit,
        ),
        "weak_sectors": _frame_records(
            weak_sectors,
            ["name", "change_pct", "up_count", "down_count", "leader", "leader_change_pct"],
            limit=sector_limit,
        ),
        "strong_stocks": _frame_records(
            strong_stocks,
            ["code", "name", "price", "change_pct", "turnover", "turnover_rate", "pe_ttm"],
            limit=stock_limit,
        ),
        "weak_stocks": _frame_records(
            weak_stocks,
            ["code", "name", "price", "change_pct", "turnover", "turnover_rate", "pe_ttm"],
            limit=stock_limit,
        ),
        "key_news": _frame_records(
            news,
            ["source", "title", "published_at"],
            limit=news_limit,
        ),
    }


def context_preview(context: dict[str, Any]) -> str:
    """以紧凑 JSON 形式预览将要送入模型的上下文。"""
    return json.dumps(context, ensure_ascii=False, indent=2)


def generate_postmarket_report(context: dict[str, Any], config: DeepSeekConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """调用 DeepSeek 生成结构化盘后复盘日报。"""
    prompt = (
        "请根据以下 context json 生成一份 A 股盘后复盘日报，"
        "重点回答市场风格、主线方向、主要催化、明日观察点和风险点。\n\n"
        f"context json:\n{context_preview(context)}"
    )
    report, meta = _call_deepseek_structured_compatible(
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        config=config,
        tool_name="submit_postmarket_report",
        tool_description="提交 A 股盘后复盘日报的结构化结果。",
        schema=REPORT_SCHEMA,
    )
    return _normalize_report(report), meta


def answer_market_question(
    context: dict[str, Any],
    question: str,
    history: list[dict[str, str]],
    config: DeepSeekConfig,
) -> tuple[str, dict[str, Any]]:
    """结合盘后上下文与多轮对话，回答用户追问。"""
    if not question.strip():
        raise AIServiceError("问题为空，无法发起 AI 问答。")

    messages: list[dict[str, str]] = [{"role": "system", "content": QNA_SYSTEM_PROMPT}]
    for item in history[-6:]:
        role = item.get("role", "")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": (
                "以下是今日盘后复盘上下文 json，请结合它回答问题。\n\n"
                f"context json:\n{context_preview(context)}\n\n"
                f"用户问题：{question}"
            ),
        }
    )

    try:
        answer_payload, meta = _call_deepseek_structured_compatible(
            messages=messages,
            config=config,
            tool_name="submit_market_answer",
            tool_description="提交盘后分析问答的结构化回答。",
            schema=QNA_SCHEMA,
        )
        return _format_structured_answer(answer_payload), meta
    except AIServiceError:
        payload = _call_deepseek(messages=messages, config=config, temperature=0)
        message = _first_message(payload)
        content = (message.get("content") or "").strip()
        if not content:
            raise AIServiceError("DeepSeek 未返回有效答复，请稍后再试。")
        meta = _response_meta(payload)
        meta["structured_mode"] = "plain_text_fallback"
        return content, meta


def _call_deepseek_structured(
    messages: list[dict[str, str]],
    config: DeepSeekConfig,
    tool_name: str,
    tool_description: str,
    schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """优先使用 DeepSeek strict function calling 获取结构化结果。"""
    payload = _call_deepseek(
        messages=messages,
        config=config,
        base_url=DEEPSEEK_BETA_BASE_URL,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "strict": True,
                    "parameters": schema,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        temperature=0,
    )
    arguments = _extract_tool_arguments(payload, tool_name)
    if not isinstance(arguments, dict):
        raise AIServiceError("strict function calling 未返回合法的 JSON arguments。")
    return arguments, payload


def _call_deepseek_structured_compatible(
    messages: list[dict[str, str]],
    config: DeepSeekConfig,
    tool_name: str,
    tool_description: str,
    schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """按 DeepSeek 能力限制选择可用的结构化输出方式。

    DeepSeek 的 thinking 模式当前不支持 `tool_choice`，因此开启思考时
    直接使用 JSON Output；关闭思考时才优先使用 strict function calling。
    """
    if config.thinking_enabled:
        data, payload = _call_deepseek_json_fallback(messages=messages, config=config)
        meta = _response_meta(payload)
        meta["structured_mode"] = "json_output_thinking"
        return data, meta

    try:
        data, payload = _call_deepseek_structured(
            messages=messages,
            config=config,
            tool_name=tool_name,
            tool_description=tool_description,
            schema=schema,
        )
        meta = _response_meta(payload)
        meta["structured_mode"] = "strict_function_call"
        return data, meta
    except AIServiceError as strict_exc:
        data, payload = _call_deepseek_json_fallback(messages=messages, config=config)
        meta = _response_meta(payload)
        meta["structured_mode"] = "json_fallback"
        meta["structured_fallback_reason"] = str(strict_exc)
        return data, meta


def _call_deepseek_json_fallback(
    messages: list[dict[str, str]],
    config: DeepSeekConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """使用 JSON Output 模式获取结构化结果。"""
    payload = _call_deepseek(
        messages=messages,
        config=config,
        response_format={"type": "json_object"},
        temperature=0,
    )
    message = _first_message(payload)
    content = (message.get("content") or "").strip()
    if not content:
        raise AIServiceError("DeepSeek 返回了空内容。JSON Output 模式未得到可解析结果。")

    try:
        parsed = json.loads(_extract_json_text(content))
    except json.JSONDecodeError as exc:
        raise AIServiceError(f"JSON Output 模式返回内容无法解析：{exc}") from exc

    if not isinstance(parsed, dict):
        raise AIServiceError("JSON Output 模式返回的结果不是对象。")
    return parsed, payload


def _call_deepseek(
    messages: list[dict[str, str]],
    config: DeepSeekConfig,
    base_url: str = DEEPSEEK_BASE_URL,
    response_format: dict[str, str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """向 DeepSeek Chat Completions 接口发起请求。"""
    if not config.api_key.strip():
        raise AIServiceError("尚未配置 DeepSeek API Key。")

    payload = _build_deepseek_payload(
        messages=messages,
        config=config,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
    )

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=config.timeout,
        )
    except requests.RequestException as exc:
        raise AIServiceError(f"DeepSeek 请求失败：{exc}") from exc

    if response.status_code >= 400:
        detail = _error_detail(response)
        raise AIServiceError(f"DeepSeek 接口返回 {response.status_code}：{detail}")

    try:
        return response.json()
    except ValueError as exc:
        raise AIServiceError("DeepSeek 返回的结果不是合法 JSON。") from exc


def _build_deepseek_payload(
    messages: list[dict[str, str]],
    config: DeepSeekConfig,
    response_format: dict[str, str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """集中构建 Chat Completions 请求体，便于后续新增参数。"""
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "max_tokens": config.max_tokens,
        "thinking": {"type": "enabled" if config.thinking_enabled else "disabled"},
        "temperature": config.temperature if temperature is None else temperature,
        "stream": False,
    }
    if config.thinking_enabled:
        payload["reasoning_effort"] = config.reasoning_effort

    optional_fields = {
        "response_format": response_format,
        "tools": tools,
        "tool_choice": tool_choice,
    }
    payload.update({key: value for key, value in optional_fields.items() if value is not None})
    return payload


def _first_message(payload: dict[str, Any]) -> dict[str, Any]:
    """读取第一条候选消息。"""
    choices = payload.get("choices") or []
    if not choices:
        raise AIServiceError("DeepSeek 未返回任何候选结果。")
    message = choices[0].get("message") or {}
    if not message:
        raise AIServiceError("DeepSeek 返回内容缺少 message 字段。")
    return message


def _extract_tool_arguments(payload: dict[str, Any], tool_name: str) -> dict[str, Any]:
    """从 tool_calls 中解析指定函数的 arguments。"""
    message = _first_message(payload)
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        raise AIServiceError("模型没有返回 tool_calls。")

    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        if function.get("name") != tool_name:
            continue
        raw_arguments = function.get("arguments")
        if not raw_arguments:
            raise AIServiceError("tool_calls 中缺少 arguments。")
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise AIServiceError(f"tool_call arguments 无法解析：{exc}") from exc
        if not isinstance(parsed, dict):
            raise AIServiceError("tool_call arguments 不是对象。")
        return parsed

    raise AIServiceError(f"未找到名为 {tool_name} 的 tool_call。")


def _frame_records(frame: pd.DataFrame, columns: list[str], limit: int) -> list[dict[str, Any]]:
    """从 DataFrame 中提取更适合模型消费的小样本记录。"""
    if frame.empty:
        return []

    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return []

    scoped = frame[existing].head(limit).copy()
    records: list[dict[str, Any]] = []
    for record in scoped.to_dict(orient="records"):
        records.append({key: _to_scalar(value) for key, value in record.items()})
    return records


def _to_scalar(value: Any) -> Any:
    """把 pandas/numpy 标量转换成普通 Python 值。"""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value


def _extract_json_text(text: str) -> str:
    """兼容模型偶尔返回 markdown 代码块的情况。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return stripped[start : end + 1]
    return stripped


def _normalize_report(report: dict[str, Any]) -> dict[str, Any]:
    """兜底补齐前端展示需要的字段。"""
    return {
        "market_tone": str(report.get("market_tone", "待判断")),
        "summary": str(report.get("summary", "暂无总结")),
        "key_points": _ensure_string_list(report.get("key_points")),
        "sector_views": _ensure_sector_views(report.get("sector_views")),
        "news_watch": _ensure_news_watch(report.get("news_watch")),
        "next_focus": _ensure_string_list(report.get("next_focus")),
        "risk_flags": _ensure_string_list(report.get("risk_flags")),
        "disclaimer": str(report.get("disclaimer", "以上内容仅用于复盘整理，不构成投资建议。")),
    }


def _normalize_qna_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """兜底补齐问答结构化结果。"""
    return {
        "conclusion": str(payload.get("conclusion", "当前上下文不足以确认。")),
        "evidence": _ensure_string_list(payload.get("evidence")),
        "uncertainty": str(payload.get("uncertainty", "")),
        "risk_tip": str(payload.get("risk_tip", "以上内容仅用于复盘整理，不构成投资建议。")),
    }


def _format_structured_answer(payload: dict[str, Any]) -> str:
    """把结构化问答结果格式化为适合聊天窗口展示的文本。"""
    normalized = _normalize_qna_payload(payload)
    sections = [normalized["conclusion"]]
    if normalized["evidence"]:
        evidence_block = "\n".join(f"- {item}" for item in normalized["evidence"])
        sections.append(f"依据：\n{evidence_block}")
    if normalized["uncertainty"]:
        sections.append(f"不确定性：{normalized['uncertainty']}")
    if normalized["risk_tip"]:
        sections.append(f"风险提示：{normalized['risk_tip']}")
    return "\n\n".join(section for section in sections if section.strip())


def _ensure_string_list(value: Any) -> list[str]:
    """将任意值归一为字符串列表。"""
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _ensure_sector_views(value: Any) -> list[dict[str, str]]:
    """标准化板块观点结构。"""
    views: list[dict[str, str]] = []
    if not isinstance(value, list):
        return views
    for item in value:
        if not isinstance(item, dict):
            continue
        views.append(
            {
                "sector": str(item.get("sector", "未命名板块")),
                "view": str(item.get("view", "")),
                "driver": str(item.get("driver", "")),
                "risk": str(item.get("risk", "")),
            }
        )
    return views


def _ensure_news_watch(value: Any) -> list[dict[str, str]]:
    """标准化消息观察结构。"""
    items: list[dict[str, str]] = []
    if not isinstance(value, list):
        return items
    for item in value:
        if not isinstance(item, dict):
            continue
        sentiment = str(item.get("sentiment", "中性"))
        items.append(
            {
                "title": str(item.get("title", "未命名消息")),
                "impact": str(item.get("impact", "")),
                "sentiment": sentiment if sentiment in VALID_NEWS_SENTIMENTS else "中性",
            }
        )
    return items


def _response_meta(payload: dict[str, Any]) -> dict[str, Any]:
    """提取便于页面展示的响应元信息。"""
    usage = payload.get("usage") or {}
    return {
        "model": payload.get("model", ""),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _error_detail(response: requests.Response) -> str:
    """尽量从错误响应中提取更友好的提示文本。"""
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or "未知错误"

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("type") or error.get("code")
        return str(message)
    if error:
        return str(error)
    return response.text.strip() or "未知错误"
