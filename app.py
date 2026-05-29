from __future__ import annotations

# 股市复盘助手的 Streamlit 前端页面。
# 页面按复盘流程拆成六个标签页：
# 1. 大盘概览：先看指数和市场宽度。
# 2. 板块热力：观察哪些方向领涨、哪些方向走弱。
# 3. 个股异动：快速筛查强弱个股和异常成交。
# 4. 财经快讯：集中复盘东方财富和同花顺快讯。
# 5. 盘后分析：使用 DeepSeek 生成 AI 复盘日报与答疑。
# 6. 复盘记忆库：查看历史复盘、连续性跟踪并支持删除记录。

import streamlit as st

from stock_assistant.ai_insights import (
    AIServiceError,
    DeepSeekConfig,
    answer_market_question,
    build_analysis_context,
    context_preview,
    generate_postmarket_report,
)
from stock_assistant.market import load_market_snapshot, load_sector_history, market_breadth, top_rows
from stock_assistant.memory import (
    build_review_memory_entry,
    delete_review_memory_entries,
    load_review_memory,
    memory_detail_frame,
    memory_timeline_frame,
    review_memory_overview,
    save_review_memory_entry,
)
from stock_assistant.news import collect_news, load_sources
from stock_assistant.settings import (
    AI_MODELS,
    AI_REASONING_EFFORTS,
    MEMORY_STORE,
    NEWS_CONFIG,
    PAGE_STYLE,
    SECTOR_HISTORY_WINDOWS,
)
from stock_assistant.ui_helpers import (
    ai_context_signature,
    default_deepseek_api_key,
    existing_columns,
    format_number,
    format_pct,
    format_structured_mode,
    metric_delta,
    render_report_section,
    sector_selection_token,
    selected_sector_name,
    trim_sector_history,
)
from stock_assistant.visualizations import sector_bar, sector_history_chart, sector_treemap, stock_scatter


# 宽屏布局更适合图表和大表格展示，减少信息拥挤。
st.set_page_config(page_title="股市分析助手", page_icon="📈", layout="wide")
st.markdown(PAGE_STYLE, unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def cached_market():
    """返回当前会话中的行情缓存快照。"""
    # 行情变化更快，缓存时间短一些，减少重复请求但保留刷新感。
    return load_market_snapshot()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_news():
    """返回基于配置渠道构建的财经快讯缓存表。"""
    # 快讯更新频率低于盘口数据但高于公告类消息，缓存时间折中处理。
    return collect_news(load_sources(NEWS_CONFIG))


@st.cache_data(ttl=21600, show_spinner=False)
def cached_sector_history(name: str, lookback_days: int = 400):
    """缓存单个板块的历史趋势数据。"""
    return load_sector_history(name, lookback_days)


@st.cache_data(ttl=30, show_spinner=False)
def cached_review_memory(path: str):
    """缓存本地复盘记忆库，避免频繁重复读取。"""
    return load_review_memory(path)


@st.dialog("板块历史趋势", width="large")
def show_sector_history_dialog(sector_name: str) -> None:
    """弹窗展示板块历史涨跌趋势，并支持常见时间窗口切换。"""
    st.caption(f"{sector_name} 的历史涨跌趋势")
    window_label = st.radio(
        "时间窗口",
        options=list(SECTOR_HISTORY_WINDOWS.keys()),
        index=2,
        horizontal=True,
        key="sector_history_window",
        label_visibility="collapsed",
    )

    with st.spinner("正在加载历史趋势..."):
        history, source, error = cached_sector_history(sector_name, 400)

    if history.empty:
        st.warning("暂时没有拿到这个板块的历史趋势数据。")
        if error:
            st.caption(error)
        if st.button("关闭", type="primary", use_container_width=True):
            st.rerun()
        return

    scoped_history = trim_sector_history(history, SECTOR_HISTORY_WINDOWS[window_label])
    latest_row = scoped_history.iloc[-1]
    latest_close = latest_row.get("close")
    latest_change = latest_row.get("change_pct")
    window_return = scoped_history["cum_return_pct"].iloc[-1]
    high_value = scoped_history["high"].max() if "high" in scoped_history else scoped_history["close"].max()
    up_days = int((scoped_history["change_pct"] > 0).sum()) if "change_pct" in scoped_history else 0

    metric_cols = st.columns(4)
    metric_cols[0].metric("区间涨跌", format_pct(window_return))
    metric_cols[1].metric("最新点位", format_number(latest_close), metric_delta(latest_change), delta_color="inverse")
    metric_cols[2].metric("区间高点", format_number(high_value))
    metric_cols[3].metric("上涨天数", f"{up_days}/{len(scoped_history)}")

    st.plotly_chart(sector_history_chart(scoped_history, sector_name), use_container_width=True)
    if source == "演示数据":
        st.info("当前展示的是演示趋势曲线，说明真实历史接口暂时不可用。")
    st.caption(f"数据来源：{source}")
    if error and source != "演示数据":
        st.caption(error)
    if st.button("关闭", type="primary", use_container_width=True):
        st.rerun()


# 在页面顶部统一取数，保证所有标签页基于同一批快照数据。
snapshot = cached_market()
news = cached_news()

# 页面头部先告诉用户数据来源和更新时间，方便判断复盘时效性。
st.title("股市分析助手")
st.caption(f"数据来源：{snapshot.source} ｜ 更新时间：{snapshot.fetched_at:%Y-%m-%d %H:%M:%S}")
if "演示数据" in snapshot.source:
    reason = snapshot.error or "未知原因"
    st.warning(f"部分行情未成功加载真实数据，已局部回退到演示数据。失败原因：{reason}")

with st.sidebar:
    # 侧边栏承载全局复盘参数，所有标签页都会受这里的设置影响。
    st.header("复盘设置")
    sector_count = st.slider("板块排行数量", 5, 30, 8)
    stock_count = st.slider("个股排行数量", 5, 50, 10)
    
    st.divider()
    st.header("AI 设置")
    ai_api_key = default_deepseek_api_key()
    if ai_api_key.strip():
        st.success("已检测到环境变量 `DEEPSEEK_API_KEY`。")
    else:
        st.warning("尚未检测到环境变量 `DEEPSEEK_API_KEY`，AI 功能当前不可用。")
    ai_model = st.selectbox("AI 模型", AI_MODELS, index=0)
    ai_thinking_enabled = st.toggle("启用思考模式", value=True, help="盘后分析更适合启用思考模式，回答会更稳。")
    ai_reasoning_effort = st.select_slider(
        "推理强度",
        options=AI_REASONING_EFFORTS,
        value="high",
        disabled=not ai_thinking_enabled,
        help="仅在启用思考模式时生效。",
    )
    ai_max_tokens = st.slider("AI 最大输出 Token", 800, 4000, 1800, 100)

    st.divider()
    if st.button("刷新缓存", use_container_width=True):
        # 用户主动刷新时，清空缓存并重新拉取行情与消息。
        st.cache_data.clear()
        st.rerun()

# 标签页顺序按典型盘后复盘流程组织：先大盘，再板块，再个股，再消息，最后 AI 与记忆库视图。
overview_tab, sector_tab, stock_tab, news_tab, ai_tab, memory_tab = st.tabs(
    ["大盘概览", "板块热力", "个股异动", "财经快讯", "盘后分析", "复盘记忆库"]
)
analysis_context = build_analysis_context(
    snapshot,
    news,
    sector_limit=min(sector_count, 8),
    stock_limit=min(stock_count, 8),
    news_limit=12,
)
analysis_signature = ai_context_signature(analysis_context)
ai_config = DeepSeekConfig(
    api_key=ai_api_key,
    model=ai_model,
    thinking_enabled=ai_thinking_enabled,
    reasoning_effort=ai_reasoning_effort,
    max_tokens=ai_max_tokens,
)

with overview_tab:
    # 先把全市场涨跌分布算出来，供顶部指标卡直接复用。
    breadth = market_breadth(snapshot.stocks)
    # 顶部指标卡用于快速判断当天市场整体强弱。
    cols = st.columns(5)
    cols[0].metric("上涨家数", breadth["up"])
    cols[1].metric("下跌家数", breadth["down"])
    cols[2].metric("平盘家数", breadth["flat"])
    cols[3].metric("涨停附近", breadth["limit_up"])
    cols[4].metric("跌停附近", breadth["limit_down"])

    st.subheader("主要指数")
    # 每个主要指数单独占一个指标卡，方便横向比较。
    index_cols = st.columns(len(snapshot.indices) if not snapshot.indices.empty else 1)
    for idx, row in snapshot.indices.iterrows():
        index_cols[idx % len(index_cols)].metric(
            row.get("name", "-"),
            f"{row.get('price', 0):,.2f}",
            metric_delta(row.get("change_pct")),
            delta_color="inverse",
        )

    # 板块涨跌放在概览页中下方，并占满整行，方便横向扫读板块强弱。
    sorted_overview_sectors = snapshot.sectors.sort_values("change_pct", ascending=False).reset_index(drop=True)
    sector_min = float(sorted_overview_sectors["change_pct"].min()) if not sorted_overview_sectors.empty else 0.0
    sector_max = float(sorted_overview_sectors["change_pct"].max()) if not sorted_overview_sectors.empty else 0.0
    sector_padding = max((sector_max - sector_min) * 0.08, 0.4)
    sector_y_range = [min(0.0, sector_min) - sector_padding, max(0.0, sector_max) + sector_padding]
    sector_chart_height = max(520, min(650, len(sorted_overview_sectors) * 18 + 360))
    st.subheader("板块涨跌")
    st.plotly_chart(
        sector_bar(sorted_overview_sectors, y_range=sector_y_range, height=sector_chart_height),
        use_container_width=True,
    )

with sector_tab:
    # 热力图同时表达板块体量和当天表现，适合看资金集中方向。
    st.subheader("行业/概念板块热力")
    st.caption("点击热力图中的板块，可弹出查看历史涨跌趋势，并切换常见时间窗口。")
    should_open_history_dialog = False
    active_sector = None
    treemap_nonce = st.session_state.get("sector_treemap_nonce", 0)
    treemap_placeholder = st.empty()
    treemap_event = treemap_placeholder.plotly_chart(
        sector_treemap(snapshot.sectors),
        use_container_width=True,
        key=f"sector_treemap_chart_{treemap_nonce}",
        on_select="rerun",
        selection_mode="points",
    )
    selected_sector = selected_sector_name(treemap_event)
    selection_token = sector_selection_token(treemap_event, treemap_nonce)
    if selected_sector:
        st.session_state["last_selected_sector"] = selected_sector
        if selection_token and st.session_state.get("last_sector_selection_token") != selection_token:
            st.session_state["last_sector_selection_token"] = selection_token
            st.session_state["sector_treemap_nonce"] = treemap_nonce + 1
            treemap_placeholder.empty()
            treemap_placeholder.plotly_chart(
                sector_treemap(snapshot.sectors),
                use_container_width=True,
                key=f"sector_treemap_chart_{treemap_nonce + 1}",
                on_select="rerun",
                selection_mode="points",
            )
            should_open_history_dialog = True
            active_sector = selected_sector

    remembered_sector = st.session_state.get("last_selected_sector")
    if remembered_sector:
        hint_col, button_col = st.columns([1, 0.34], vertical_alignment="bottom")
        with hint_col:
            st.caption(f"当前已选板块：{remembered_sector}")
        with button_col:
            if st.button(f"查看 {remembered_sector} 历史趋势", key="reopen_sector_history", use_container_width=True):
                should_open_history_dialog = True
                active_sector = remembered_sector

    if should_open_history_dialog and active_sector:
        show_sector_history_dialog(active_sector)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**领涨板块**")
        st.dataframe(
            top_rows(snapshot.sectors, "change_pct", sector_count)[
                existing_columns(snapshot.sectors, ["name", "change_pct", "up_count", "down_count", "leader", "leader_change_pct"])
            ],
            use_container_width=True,
            hide_index=True,
        )
    with col2:
        st.markdown("**领跌板块**")
        st.dataframe(
            top_rows(snapshot.sectors, "change_pct", sector_count, ascending=True)[
                existing_columns(snapshot.sectors, ["name", "change_pct", "up_count", "down_count", "leader", "leader_change_pct"])
            ],
            use_container_width=True,
            hide_index=True,
        )

with stock_tab:
    # 散点图更容易发现“涨幅大但换手异常”或“高换手但走弱”的个股。
    st.subheader("个股异动分布")
    st.plotly_chart(stock_scatter(snapshot.stocks), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**强势个股**")
        st.dataframe(
            top_rows(snapshot.stocks, "change_pct", stock_count)[
                existing_columns(snapshot.stocks, ["code", "name", "price", "change_pct", "turnover", "turnover_rate", "pe_ttm"])
            ],
            use_container_width=True,
            hide_index=True,
        )
    with col2:
        st.markdown("**弱势个股**")
        st.dataframe(
            top_rows(snapshot.stocks, "change_pct", stock_count, ascending=True)[
                existing_columns(snapshot.stocks, ["code", "name", "price", "change_pct", "turnover", "turnover_rate", "pe_ttm"])
            ],
            use_container_width=True,
            hide_index=True,
        )

with news_tab:
    st.subheader("消息快讯池")
    st.caption("当日消息展示")

    if news.empty:
        st.info("当前暂无当日消息。")
    else:
        st.dataframe(
            news[["source", "title", "published_at"]],
            height=900,
            use_container_width=True,
            hide_index=True,
        )

with ai_tab:
    st.subheader("AI 盘后分析")
    st.caption("这个页面聚合 DeepSeek 驱动的盘后复盘日报、重点观察与智能答疑。")

    st.session_state.setdefault("ai_report_data", None)
    st.session_state.setdefault("ai_report_meta", {})
    st.session_state.setdefault("ai_report_signature", "")
    st.session_state.setdefault("ai_report_error", "")
    st.session_state.setdefault("ai_chat_history", [])
    st.session_state.setdefault("ai_chat_meta", {})
    st.session_state.setdefault("ai_question_input", "")
    st.session_state.setdefault("review_memory_notice", "")

    report_col, qa_col = st.columns([1.08, 0.92], vertical_alignment="top")

    with report_col:
        st.markdown("**AI 盘后日报**")
        st.caption("基于今天的指数、板块、个股异动和快讯列表，自动整理出结构化复盘结论。")
        report = st.session_state.get("ai_report_data")
        current_report_meta = st.session_state.get("ai_report_meta", {})
        current_memory_entry = (
            build_review_memory_entry(
                snapshot,
                report,
                report_meta=current_report_meta,
            )
            if report
            else None
        )
        has_report = report is not None

        control_cols = st.columns([0.42, 0.26, 0.32])
        if control_cols[0].button("生成 AI 盘后日报", type="primary", use_container_width=True):
            if not ai_config.api_key.strip():
                st.session_state["ai_report_error"] = "尚未检测到环境变量 DEEPSEEK_API_KEY，请先在系统环境变量中配置后再生成。"
            else:
                st.session_state["ai_report_error"] = ""
                with st.spinner("DeepSeek 正在整理盘后复盘日报..."):
                    try:
                        report, meta = generate_postmarket_report(analysis_context, ai_config)
                    except AIServiceError as exc:
                        st.session_state["ai_report_error"] = str(exc)
                    else:
                        st.session_state["ai_report_data"] = report
                        st.session_state["ai_report_meta"] = meta
                        st.session_state["ai_report_signature"] = analysis_signature
                        st.rerun()

        if control_cols[1].button("写入复盘记忆", use_container_width=True, disabled=not has_report):
            if current_memory_entry is not None:
                action = save_review_memory_entry(MEMORY_STORE, current_memory_entry)
                cached_review_memory.clear()
                st.session_state["review_memory_notice"] = "已新增到复盘记忆库。" if action == "created" else "已更新当日复盘记忆。"
                st.rerun()

        if control_cols[2].button("清空 AI 结果", use_container_width=True):
            st.session_state["ai_report_data"] = None
            st.session_state["ai_report_meta"] = {}
            st.session_state["ai_report_signature"] = ""
            st.session_state["ai_report_error"] = ""

        if st.session_state["ai_report_error"]:
            st.warning(st.session_state["ai_report_error"])

        report = st.session_state.get("ai_report_data")
        report_signature = st.session_state.get("ai_report_signature", "")
        if report and report_signature and report_signature != analysis_signature:
            st.info("当前 AI 报告基于旧的行情或快讯数据生成。若你刚刷新了数据，建议重新生成一次。")

        if st.session_state.get("review_memory_notice"):
            st.success(st.session_state["review_memory_notice"])
            st.session_state["review_memory_notice"] = ""

        with st.container(height=650, border=True):
            if report:
                summary_cols = st.columns([0.34, 0.66], vertical_alignment="top")
                with summary_cols[0]:
                    st.metric("市场风格判断", report.get("market_tone", "待判断"))
                    meta = st.session_state.get("ai_report_meta", {})
                    if meta.get("total_tokens"):
                        st.caption(
                            f"模型：{meta.get('model') or ai_config.model} ｜ Tokens：{meta.get('total_tokens')}"
                        )
                    structured_mode = format_structured_mode(meta.get("structured_mode"))
                    if structured_mode:
                        st.caption(f"结构化模式：{structured_mode}")
                with summary_cols[1]:
                    st.markdown("**日报摘要**")
                    st.write(report.get("summary", "暂无摘要"))

                render_report_section("关键结论", report.get("key_points", []), "暂无关键结论。")

                st.markdown("**板块观点**")
                sector_views = report.get("sector_views", [])
                if not sector_views:
                    st.caption("暂无板块观点。")
                else:
                    for item in sector_views:
                        with st.container(border=True):
                            st.markdown(f"**{item['sector']}**")
                            st.write(item["view"] or "暂无观点")
                            st.caption(f"驱动：{item['driver'] or '待补充'} ｜ 风险：{item['risk'] or '待补充'}")

                st.markdown("**消息影响梳理**")
                news_watch = report.get("news_watch", [])
                if not news_watch:
                    st.caption("暂无消息影响梳理。")
                else:
                    for item in news_watch:
                        with st.container(border=True):
                            st.markdown(f"**{item['sentiment']}｜{item['title']}**")
                            st.write(item["impact"] or "暂无影响说明")

                render_report_section("明日观察重点", report.get("next_focus", []), "暂无明日观察重点。")
                render_report_section("风险提示", report.get("risk_flags", []), "暂无额外风险提示。")
                st.caption(report.get("disclaimer", "以上内容仅用于复盘整理，不构成投资建议。"))
            else:
                st.info("当前还没有生成 AI 盘后日报。配置好环境变量 `DEEPSEEK_API_KEY` 后，点击上方按钮即可开始。")

        with st.expander("查看将发送给 DeepSeek 的上下文摘要"):
            st.code(context_preview(analysis_context), language="json")

    with qa_col:
        st.markdown("**AI 盘后答疑**")
        st.caption("你可以直接追问“今天主线是什么”“哪些消息更值得明天继续跟踪”等问题。")
        suggestion_cols = st.columns(3)
        suggestion_texts = ["今天的主线方向是什么？", "哪些消息对明天影响更大？", "当前最大的风险点是什么？"]
        for idx, text in enumerate(suggestion_texts):
            if suggestion_cols[idx].button(text, key=f"ai_question_suggestion_{idx}", use_container_width=True):
                st.session_state["ai_question_input"] = text

        with st.container(height=450, border=True):
            chat_history = st.session_state.get("ai_chat_history", [])
            if not chat_history:
                st.info("当前还没有问答记录。你可以先点击上方推荐问题，或在下方直接输入。")
            else:
                for item in chat_history:
                    with st.chat_message("user" if item["role"] == "user" else "assistant"):
                        st.markdown(item["content"])

        with st.form("ai_qna_form", clear_on_submit=False):
            st.text_area(
                "向 AI 追问",
                key="ai_question_input",
                height=100,
                placeholder="例如：今天指数修复，但主线持续性够不够？",
            )
            form_action_cols = st.columns([0.72, 0.28])
            submitted = form_action_cols[0].form_submit_button(
                "发送问题",
                type="primary",
                use_container_width=True,
            )
            clear_chat = form_action_cols[1].form_submit_button("清空对话", use_container_width=True)

        if clear_chat:
            st.session_state["ai_chat_history"] = []
            st.session_state["ai_chat_meta"] = {}
            st.rerun()

        if submitted:
            question = st.session_state.get("ai_question_input", "").strip()
            if not question:
                st.warning("先输入一个问题再发送。")
            elif not ai_config.api_key.strip():
                st.warning("尚未检测到环境变量 `DEEPSEEK_API_KEY`，请先完成系统环境变量配置后再提问。")
            else:
                with st.spinner("DeepSeek 正在思考你的问题..."):
                    try:
                        answer, meta = answer_market_question(
                            analysis_context,
                            question,
                            st.session_state.get("ai_chat_history", []),
                            ai_config,
                        )
                    except AIServiceError as exc:
                        st.warning(str(exc))
                    else:
                        st.session_state["ai_chat_history"].append({"role": "user", "content": question})
                        st.session_state["ai_chat_history"].append({"role": "assistant", "content": answer})
                        st.session_state["ai_chat_meta"] = meta
                        st.rerun()

        chat_meta = st.session_state.get("ai_chat_meta", {})
        if chat_meta.get("total_tokens"):
            st.caption(
                f"最近一次问答模型：{chat_meta.get('model') or ai_config.model} ｜ Tokens：{chat_meta.get('total_tokens')}"
            )
        structured_mode = format_structured_mode(chat_meta.get("structured_mode"))
        if structured_mode:
            st.caption(f"问答结构化模式：{structured_mode}")

with memory_tab:
    st.subheader("复盘记忆库")
    st.caption("统一查看历史复盘、跨日主线延续和风险累积，并支持按所选记录删除记忆。")

    st.session_state.setdefault("memory_delete_notice", "")
    st.session_state.setdefault("memory_selected_dates", [])
    st.session_state.setdefault("memory_pending_delete_dates", [])
    st.session_state.setdefault("memory_detail_date", "")
    st.session_state.setdefault("memory_clear_selection", False)

    if st.session_state.get("memory_clear_selection"):
        st.session_state.pop("memory_selected_dates", None)
        st.session_state["memory_clear_selection"] = False

    current_report = st.session_state.get("ai_report_data")
    current_report_meta = st.session_state.get("ai_report_meta", {})
    current_memory_entry = (
        build_review_memory_entry(
            snapshot,
            current_report,
            report_meta=current_report_meta,
        )
        if current_report
        else None
    )

    memory_entries = cached_review_memory(str(MEMORY_STORE))
    memory_overview = review_memory_overview(memory_entries, current_entry=current_memory_entry)
    continuity = memory_overview.get("continuity", {})

    top_cols = st.columns([0.75, 0.25], vertical_alignment="bottom")
    with top_cols[0]:
        memory_metric_cols = st.columns([0.1, 0.2, 0.1, 0.6])
        memory_metric_cols[0].metric("已保存复盘日", memory_overview.get("saved_days", 0))
        memory_metric_cols[1].metric("最近记录日期", memory_overview.get("latest_date", "-"))
        memory_metric_cols[2].metric("高频主线", memory_overview.get("dominant_sector", "暂无"))
        memory_metric_cols[3].metric("风格变化", continuity.get("tone_change", "") or "暂无")
    with top_cols[1]:
        delete_options = [
            f"{entry.get('memory_date', '')}｜{entry.get('market_tone', '待判断')}"
            for entry in memory_entries
        ]
        selected_labels = st.multiselect(
            "选择要删除的复盘记忆",
            options=delete_options,
            key="memory_selected_dates",
            placeholder="可多选",
        )
        selected_dates = [label.split("｜", 1)[0] for label in selected_labels]
        if st.button("删除选中记忆", use_container_width=True, disabled=not selected_dates):
            st.session_state["memory_pending_delete_dates"] = selected_dates
            st.rerun()

    pending_delete_dates = st.session_state.get("memory_pending_delete_dates", [])
    if pending_delete_dates:
        st.warning(f"即将删除以下复盘记忆：{'、'.join(pending_delete_dates)}。删除后不可恢复，请确认。")
        confirm_cols = st.columns([0.3, 0.3, 0.4])
        if confirm_cols[0].button("确认删除", type="primary", use_container_width=True):
            removed_count = delete_review_memory_entries(MEMORY_STORE, pending_delete_dates)
            cached_review_memory.clear()
            st.session_state["memory_clear_selection"] = True
            st.session_state["memory_pending_delete_dates"] = []
            st.session_state["memory_delete_notice"] = (
                f"已删除 {removed_count} 条复盘记忆。" if removed_count else "没有删除任何记录。"
            )
            st.rerun()
        if confirm_cols[1].button("取消删除", use_container_width=True):
            st.session_state["memory_pending_delete_dates"] = []
            st.rerun()

    if st.session_state.get("memory_delete_notice"):
        st.success(st.session_state["memory_delete_notice"])
        st.session_state["memory_delete_notice"] = ""

    memory_left, memory_right = st.columns([1.05, 0.95], vertical_alignment="top")

    with memory_left:
        st.markdown("**复盘时间线**")
        with st.container(height=250, border=True):
            timeline_frame = memory_timeline_frame(memory_entries, limit=20)
            if timeline_frame.empty:
                st.info("记忆库还是空的。先在“盘后分析”页生成 AI 盘后日报并写入一条记录。")
            else:
                st.dataframe(timeline_frame, use_container_width=True, hide_index=True)

        with st.expander("查看详细记忆记录"):
            detail_frame = memory_detail_frame(memory_entries, limit=20)
            if detail_frame.empty:
                st.info("暂无详细记录。")
            else:
                st.dataframe(detail_frame, use_container_width=True, hide_index=True)

        st.markdown("**单条复盘详情**")
        detail_options = [entry.get("memory_date", "") for entry in memory_entries if entry.get("memory_date")]
        selected_detail_date = ""
        if detail_options:
            if st.session_state.get("memory_detail_date") not in detail_options:
                st.session_state["memory_detail_date"] = detail_options[0]
            selected_detail_date = st.selectbox(
                "按日期查看复盘详情",
                options=detail_options,
                key="memory_detail_date",
            )
        else:
            st.session_state["memory_detail_date"] = ""
            st.caption("暂无可选复盘日期。")
        selected_detail_entry = next(
            (entry for entry in memory_entries if entry.get("memory_date") == selected_detail_date),
            None,
        )
        with st.container(height=250, border=True):
            if not selected_detail_entry:
                st.info("选择一个已保存日期后，这里会展示该日的完整复盘详情。")
            else:
                st.caption(
                    f"日期：{selected_detail_entry.get('memory_date', '-')}"
                    f" ｜ 风格：{selected_detail_entry.get('market_tone', '待判断')}"
                    f" ｜ 来源：{selected_detail_entry.get('source', '-')}"
                )
                st.markdown("**当日摘要**")
                st.write(selected_detail_entry.get("summary", "暂无摘要"))
                render_report_section("关键结论", selected_detail_entry.get("key_points", []), "暂无关键结论。")
                sector_labels = selected_detail_entry.get("mainline_sectors", [])
                render_report_section("主线板块", sector_labels, "暂无主线板块记录。")
                render_report_section("观察重点", selected_detail_entry.get("next_focus", []), "暂无观察重点。")
                render_report_section("风险提示", selected_detail_entry.get("risk_flags", []), "暂无风险提示。")

    with memory_right:
        st.markdown("**连续性洞察**")
        with st.container(height=320, border=True):
            if continuity.get("mode") == "insufficient_history":
                st.info("至少保存 1 份历史复盘，系统才会先建立连续性基线。保存满 2 份后即可开始跨日连续性跟踪。")
            elif continuity.get("mode") == "baseline_only":
                st.caption(f"基线日期：{continuity.get('current_date', '-')}")
                st.caption("当前只有 1 份已保存复盘，系统先展示基线主线与风险，待下一份记录写入后自动开启跨日比较。")
                render_report_section("基线主线板块", continuity.get("new_sectors", []), "暂无主线板块记录。")
                render_report_section("基线风险点", continuity.get("new_risks", []), "暂无风险点记录。")
            else:
                st.caption(
                    f"对比区间：{continuity.get('previous_date', '-')}"
                    f" -> {continuity.get('current_date', '-')}"
                )
                render_report_section("延续的主线板块", continuity.get("shared_sectors", []), "暂无明显延续板块。")
                render_report_section("新增的主线板块", continuity.get("new_sectors", []), "暂无新增主线板块。")
                render_report_section("延续的风险点", continuity.get("shared_risks", []), "暂无延续风险点。")
                render_report_section("新增的风险点", continuity.get("new_risks", []), "暂无新增风险点。")

        freq_cols = st.columns(3)
        recurring_sectors = memory_overview.get("recurring_sectors", [])
        recurring_risks = memory_overview.get("recurring_risks", [])
        recurring_focus = memory_overview.get("recurring_focus", [])
        with freq_cols[0]:
            st.markdown("**近10期高频板块**")
            if recurring_sectors:
                for item in recurring_sectors:
                    st.caption(f"{item['label']} × {item['count']}")
            else:
                st.caption("暂无")
        with freq_cols[1]:
            st.markdown("**近10期高频风险**")
            if recurring_risks:
                for item in recurring_risks:
                    st.caption(f"{item['label']} × {item['count']}")
            else:
                st.caption("暂无")
        with freq_cols[2]:
            st.markdown("**近10期高频观察重点**")
            if recurring_focus:
                for item in recurring_focus:
                    st.caption(f"{item['label']} × {item['count']}")
            else:
                st.caption("暂无")
