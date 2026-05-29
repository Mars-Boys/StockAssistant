# 股市分析助手

一个用于 A 股收盘后复盘的 Streamlit 工具，聚合行情、板块表现以及东方财富和同花顺快讯，帮助快速查看市场强弱与重要资讯。

## 主要功能

- 大盘概览：指数表现、涨跌家数、涨停/跌停附近数量。
- 板块热力：行业板块涨跌排行、热力图、领涨领跌信息。
- 个股异动：按涨跌幅、成交额、换手率等指标查看强弱个股。
- 财经快讯：采集东方财富和同花顺 7x24 快讯，仅展示来源、标题和发布时间。
- AI 盘后分析：接入 DeepSeek，一键生成盘后复盘日报，并支持围绕当天行情和消息继续追问。
- 复盘记忆库：将每日 AI 盘后日报写入本地记忆库，在独立界面查看连续性跟踪，并支持删除历史记录。
- 离线兜底：行情或资讯源不可用时，自动展示演示数据。

## 快速开始

```bash
conda create -n stock-assistant python=3.12
conda activate stock-assistant
pip install -r requirements.txt
streamlit run app.py
```

启动后在浏览器中打开 Streamlit 提示的本地地址即可使用。

## 项目结构

```text
.
├── app.py                         # Streamlit 页面入口
├── config/news_sources.yml        # 财经快讯渠道配置
├── requirements.txt               # 依赖列表
└── stock_assistant
    ├── ai_insights.py             # DeepSeek 接入与 AI 盘后分析
    ├── market.py                  # 行情数据加载与兜底数据
    ├── memory.py                  # 复盘记忆库与连续性跟踪
    ├── news.py                    # 消息采集
    ├── prompts                    # AI 提示词和结构化输出 schema
    │   └── market_review.py       # 盘后日报与答疑提示词
    ├── settings.py                # 应用级配置常量
    ├── ui_helpers.py              # Streamlit 页面辅助函数
    └── visualizations.py          # 图表组件
```

## 数据来源

- 行情数据优先使用 `akshare`。
- 消息来源在 `config/news_sources.yml` 中维护，当前聚合东方财富与同花顺两个综合快讯渠道。
- AI 盘后分析通过系统环境变量 `DEEPSEEK_API_KEY` 自动读取密钥。

## 免责声明

本项目仅用于投资复盘和信息整理，不构成任何投资建议。快讯列表仅做采集展示，不代表对消息影响方向的判断。
