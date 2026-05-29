from __future__ import annotations

"""仪表盘图表构建模块。

每个函数接收已经标准化好的数据，并返回可直接渲染的 Plotly 图表对象，
这样可以把展示逻辑与数据采集逻辑拆开。
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# 红涨绿跌遵循 A 股界面中的常见表达方式。
POSITIVE = "#d73027"
NEGATIVE = "#1a9850"


def sector_bar(sectors: pd.DataFrame, y_range: list[float] | None = None, height: int = 520) -> go.Figure:
    """构建板块涨跌幅的纵向柱状图。"""
    data = sectors.sort_values("change_pct", ascending=False).copy()
    colors = [POSITIVE if value >= 0 else NEGATIVE for value in data["change_pct"]]
    fig = go.Figure(
        go.Bar(
            x=data["name"],
            y=data["change_pct"],
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>涨跌幅: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=20, t=8, b=90),
        bargap=0.2,
        xaxis_title="",
        yaxis_title="涨跌幅 %",
        template="plotly_white",
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(zeroline=True, zerolinecolor="#9ca3af", ticksuffix="%")
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig


def sector_treemap(sectors: pd.DataFrame) -> go.Figure:
    """构建同时展示板块体量和当日表现的热力图。"""
    data = sectors.copy()
    # 热力图面积必须是正数，因此缺失市值时至少给一个 1 的兜底值。
    data["market_cap"] = data.get("market_cap", pd.Series([1] * len(data))).fillna(1).clip(lower=1)
    fig = px.treemap(
        data,
        path=["name"],
        values="market_cap",
        color="change_pct",
        color_continuous_scale=[NEGATIVE, "#f7f7f7", POSITIVE],
        color_continuous_midpoint=0,
        custom_data=["name", "change_pct", "leader", "leader_change_pct", "up_count", "down_count"],
    )
    fig.update_traces(
        textinfo="label",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "涨跌幅: %{customdata[1]:.2f}%<br>"
            "领涨股: %{customdata[2]}<br>"
            "领涨股涨跌幅: %{customdata[3]:.2f}%<br>"
            "上涨家数: %{customdata[4]}<br>"
            "下跌家数: %{customdata[5]}<extra></extra>"
        ),
    )
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_white",
        clickmode="event+select",
    )
    return fig


def sector_history_chart(history: pd.DataFrame, sector_name: str) -> go.Figure:
    """构建单个板块的区间累计涨跌趋势图。"""
    data = history.sort_values("date").copy()
    latest_return = float(data["cum_return_pct"].iloc[-1]) if not data.empty else 0.0
    line_color = POSITIVE if latest_return >= 0 else NEGATIVE
    fill_color = "rgba(215, 48, 39, 0.12)" if latest_return >= 0 else "rgba(26, 152, 80, 0.12)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["cum_return_pct"],
            mode="lines",
            line=dict(color=line_color, width=3),
            fill="tozeroy",
            fillcolor=fill_color,
            customdata=data[["change_pct", "close"]].round(2).to_numpy(),
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "区间涨跌: %{y:.2f}%<br>"
                "当日涨跌: %{customdata[0]:.2f}%<br>"
                "收盘点位: %{customdata[1]:.2f}<extra></extra>"
            ),
            name=sector_name,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[data["date"].iloc[-1]],
            y=[data["cum_return_pct"].iloc[-1]],
            mode="markers",
            marker=dict(size=10, color=line_color, line=dict(color="white", width=1.5)),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=12, r=12, t=18, b=8),
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="",
        yaxis_title="区间累计涨跌幅 %",
    )
    fig.update_yaxes(zeroline=True, zerolinecolor="#9ca3af", ticksuffix="%")
    return fig


def stock_scatter(stocks: pd.DataFrame) -> go.Figure:
    """构建用于观察换手率与涨跌幅关系的散点图。"""
    data, x_column, x_label = _prepare_stock_scatter_data(stocks)
    if data.empty:
        return _empty_chart("暂无可绘制的个股异动数据")

    fig = px.scatter(
        data,
        x=x_column,
        y="change_pct",
        size="turnover",
        color="change_pct",
        hover_name="name",
        custom_data=["code", "price", "turnover_rate", "turnover"],
        color_continuous_scale=[NEGATIVE, "#f7f7f7", POSITIVE],
        color_continuous_midpoint=0,
        labels={x_column: x_label, "change_pct": "涨跌幅 %"},
        size_max=18,
    )
    fig.update_traces(
        opacity=0.62,
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "代码: %{customdata[0]}<br>"
            f"{x_label}: %{{x:.2f}}<br>"
            "涨跌幅: %{y:.2f}%<br>"
            "最新价: %{customdata[1]:.2f}<br>"
            "换手率: %{customdata[2]:.2f}%<br>"
            "成交额: %{customdata[3]:.0f}<extra></extra>"
        ),
    )
    fig.update_layout(height=420, margin=dict(l=10, r=20, t=20, b=10), template="plotly_white")
    return fig


def _prepare_stock_scatter_data(stocks: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """清洗散点图数据，并在换手率缺失时自动降级到成交额横轴。"""
    if stocks.empty or "change_pct" not in stocks:
        return pd.DataFrame(), "turnover_rate", "换手率 %"

    data = stocks.copy()
    for column in ["change_pct", "turnover_rate", "turnover", "price"]:
        if column not in data:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if "code" not in data:
        data["code"] = ""
    if "name" not in data:
        data["name"] = data["code"].astype(str)

    data["turnover"] = data["turnover"].fillna(1).clip(lower=1)
    data["price"] = data["price"].fillna(0)

    turnover_rate_coverage = data["turnover_rate"].notna().mean()
    if turnover_rate_coverage >= 0.2:
        x_column = "turnover_rate"
        x_label = "换手率 %"
        scoped = data.dropna(subset=["turnover_rate", "change_pct"]).copy()
    else:
        x_column = "turnover"
        x_label = "成交额"
        scoped = data.dropna(subset=["turnover", "change_pct"]).copy()
        scoped["turnover_rate"] = scoped["turnover_rate"].fillna(0)

    return scoped, x_column, x_label


def _empty_chart(message: str) -> go.Figure:
    """返回统一样式的空状态图表。"""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(height=420, margin=dict(l=10, r=20, t=20, b=10), template="plotly_white")
    return fig
