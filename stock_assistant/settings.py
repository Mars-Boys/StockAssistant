from __future__ import annotations

"""应用级配置常量。

该模块只放不会在运行过程中改变的路径、选项和窗口参数，避免页面入口
散落硬编码值。真正的密钥、网络结果和用户选择仍由运行时读取。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEWS_CONFIG = PROJECT_ROOT / "config" / "news_sources.yml"
MEMORY_STORE = PROJECT_ROOT / "data" / "review_memory.json"

AI_MODELS = ["deepseek-v4-pro", "deepseek-v4-flash"]
AI_REASONING_EFFORTS = ["low", "medium", "high", "max"]

SECTOR_HISTORY_WINDOWS = {
    "1周": 7,
    "1月": 30,
    "3月": 90,
    "半年": 180,
    "1年": 365,
}

PAGE_STYLE = """
<style>
.block-container {
    max-width: 100%;
    padding: 3rem 5rem 2rem;
}
@media (max-width: 900px) {
    .block-container {
        padding: 1.25rem 1rem 1.5rem;
    }
}
</style>
"""
