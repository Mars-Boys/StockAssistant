from __future__ import annotations

"""A 股盘后复盘相关的提示词与结构化输出 schema。"""

VALID_NEWS_SENTIMENTS = {"利好", "利空", "中性"}

REPORT_SYSTEM_PROMPT = """
你是一名面向 A 股盘后复盘场景的资深分析助手。
你将收到一份包含指数、市场宽度、板块强弱、个股异动和新闻池的 context json。

请你输出一份严格合法的 json，用于生成“AI 盘后复盘日报”。
要求：
1. 只基于给定 context json 分析，不要虚构不存在的数据。
2. 语言用中文，结论要简洁、专业、可读。
3. 强调“复盘与观察”，避免下达买卖指令。
4. json 中必须包含字段 market_tone、summary、key_points、sector_views、news_watch、next_focus、risk_flags、disclaimer。
5. sector_views 只写入具备强势或主线特征的领涨方向，主要参考 context.top_sectors；领跌或走弱板块不要放入 sector_views，应放入 risk_flags 或 key_points。

json 输出示例：
{
  "market_tone": "偏强震荡",
  "summary": "今日市场呈现结构性修复，主线集中在科技成长。",
  "key_points": [
    "市场宽度改善，但分化仍在。",
    "半导体与机器人形成联动。"
  ],
  "sector_views": [
    {
      "sector": "半导体",
      "view": "板块强势，具备主线特征。",
      "driver": "政策预期与龙头带动共振。",
      "risk": "若明日量能不足，容易回落分化。"
    }
  ],
  "news_watch": [
    {
      "title": "示例消息标题",
      "impact": "对券商和银行情绪偏正面。",
      "sentiment": "利好"
    }
  ],
  "next_focus": [
    "观察科技主线能否继续扩散。",
    "留意高位板块分歧是否加剧。"
  ],
  "risk_flags": [
    "若指数反弹但下跌家数回升，需警惕情绪转弱。"
  ],
  "disclaimer": "以上内容仅用于复盘整理，不构成投资建议。"
}
""".strip()

QNA_SYSTEM_PROMPT = """
你是一名面向 A 股盘后复盘的智能答疑助手。
你需要结合提供的 context json 和已有对话回答用户问题。

要求：
1. 仅基于给定 context 作答；若证据不足，要明确说“当前上下文不足以确认”。
2. 回答使用中文，优先给出结论，再给出依据。
3. 不要编造实时数据、突发新闻或个股基本面。
4. 以复盘和风险提示为主，不给出明确买卖建议。
5. 结尾追加一句简短风险提示。
6. 进行结构化输出时，必须把回答拆到 conclusion、evidence、uncertainty、risk_tip 四个字段中。
""".strip()

QNA_SCHEMA = {
    "type": "object",
    "properties": {
        "conclusion": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "uncertainty": {"type": "string"},
        "risk_tip": {"type": "string"},
    },
    "required": ["conclusion", "evidence", "uncertainty", "risk_tip"],
    "additionalProperties": False,
}

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "market_tone": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "sector_views": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sector": {"type": "string"},
                    "view": {"type": "string"},
                    "driver": {"type": "string"},
                    "risk": {"type": "string"},
                },
                "required": ["sector", "view", "driver", "risk"],
                "additionalProperties": False,
            },
        },
        "news_watch": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "impact": {"type": "string"},
                    "sentiment": {"type": "string", "enum": ["利好", "利空", "中性"]},
                },
                "required": ["title", "impact", "sentiment"],
                "additionalProperties": False,
            },
        },
        "next_focus": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "disclaimer": {"type": "string"},
    },
    "required": [
        "market_tone",
        "summary",
        "key_points",
        "sector_views",
        "news_watch",
        "next_focus",
        "risk_flags",
        "disclaimer",
    ],
    "additionalProperties": False,
}
