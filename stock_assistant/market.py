from __future__ import annotations

"""行情数据加载与兜底辅助模块。

这个模块把 AkShare 的外部接口隔离开，并将它返回的不稳定字段名
转换成应用内部统一的数据结构。即使网络或数据源失效，也会提供
可预测的演示数据，保证页面仍然可浏览。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

import pandas as pd
import requests
import re


@dataclass(frozen=True)
class MarketSnapshot:
    """供前端统一使用的行情快照容器。"""

    indices: pd.DataFrame
    sectors: pd.DataFrame
    stocks: pd.DataFrame
    source: str
    fetched_at: datetime
    error: str = ""
    indices_source: str = ""
    sectors_source: str = ""
    stocks_source: str = ""


# 这些映射表把数据源的原始中文列名转换成应用内部统一使用的英文列名。

INDEX_COLUMNS = {
    "代码": "code",
    "名称": "name",
    "最新价": "price",
    "涨跌幅": "change_pct",
    "涨跌额": "change",
    "成交量": "volume",
    "成交额": "turnover",
}

STOCK_COLUMNS = {
    "代码": "code",
    "名称": "name",
    "最新价": "price",
    "涨跌幅": "change_pct",
    "涨跌额": "change",
    "成交量": "volume",
    "成交额": "turnover",
    "换手率": "turnover_rate",
    "市盈率-动态": "pe_ttm",
}

SECTOR_COLUMNS = {
    "板块名称": "name",
    "最新价": "price",
    "涨跌幅": "change_pct",
    "涨跌额": "change",
    "总市值": "market_cap",
    "换手率": "turnover_rate",
    "上涨家数": "up_count",
    "下跌家数": "down_count",
    "领涨股票": "leader",
    "领涨股票-涨跌幅": "leader_change_pct",
}

SECTOR_THS_COLUMNS = {
    "板块": "name",
    "均价": "price",
    "涨跌幅": "change_pct",
    "上涨家数": "up_count",
    "下跌家数": "down_count",
    "领涨股": "leader",
    "领涨股-涨跌幅": "leader_change_pct",
}

SECTOR_HISTORY_COLUMNS = {
    "日期": "date",
    "开盘": "open",
    "开盘价": "open",
    "收盘": "close",
    "收盘价": "close",
    "最高": "high",
    "最高价": "high",
    "最低": "low",
    "最低价": "low",
    "涨跌幅": "change_pct",
    "成交量": "volume",
    "成交额": "turnover",
    "成交值": "turnover",
}

SINA_INDEX_CODES = {
    "s_sh000001": "上证指数",
    "s_sz399001": "深证成指",
    "s_sz399006": "创业板指",
}
SINA_INDEX_LINE_PATTERN = re.compile(r'var hq_str_(?P<code>[^=]+)="(?P<body>.*)";')


def load_market_snapshot() -> MarketSnapshot:
    """加载当前 A 股行情，并允许不同数据块分别回退。"""
    indices, indices_source, indices_error = _load_indices()
    stocks, stocks_source, stocks_error = _load_stocks()
    sectors, sectors_source, sectors_error = _load_sectors()

    errors = [item for item in [indices_error, stocks_error, sectors_error] if item]
    source_parts = []
    if indices_source:
        source_parts.append(f"指数:{indices_source}")
    if stocks_source:
        source_parts.append(f"个股:{stocks_source}")
    if sectors_source:
        source_parts.append(f"板块:{sectors_source}")

    return MarketSnapshot(
        indices=_clean_numeric(indices),
        sectors=_clean_numeric(sectors),
        stocks=_clean_numeric(stocks),
        source=" ｜ ".join(source_parts) if source_parts else "演示数据",
        fetched_at=datetime.now(),
        error="；".join(errors),
        indices_source=indices_source,
        sectors_source=sectors_source,
        stocks_source=stocks_source,
    )


def load_sector_history(name: str, lookback_days: int = 400) -> tuple[pd.DataFrame, str, str]:
    """加载单个板块的历史走势，优先真实数据，失败时回退到演示曲线。"""
    lookback_days = max(lookback_days, 30)
    end_at = datetime.now()
    start_at = end_at - timedelta(days=lookback_days)
    start_date = start_at.strftime("%Y%m%d")
    end_date = end_at.strftime("%Y%m%d")

    loaders = [
        ("同花顺行业板块历史", _load_sector_history_from_ths_industry),
        ("同花顺概念板块历史", _load_sector_history_from_ths_concept),
    ]
    errors: list[str] = []
    for label, loader in loaders:
        try:
            frame = loader(name, start_date, end_date)
            if not frame.empty:
                return frame, label, ""
        except Exception as exc:
            errors.append(f"{label}失败：{_humanize_market_error(exc)}")

    return _demo_sector_history(name, lookback_days), "演示数据", "；".join(errors) or "板块历史趋势加载失败：未知原因"


def _load_indices() -> tuple[pd.DataFrame, str, str]:
    """优先加载三大指数真实数据，失败时仅回退指数本身。"""
    errors: list[str] = []
    try:
        frame = _load_major_indices_from_sina()
        if not frame.empty:
            return frame, "新浪三大指数", ""
    except Exception as exc:
        errors.append(f"新浪三大指数失败：{_humanize_market_error(exc)}")
    return _demo_indices(), "演示数据", "；".join(errors)


def _load_stocks() -> tuple[pd.DataFrame, str, str]:
    """加载个股列表，失败时仅回退个股数据。"""
    loaders = [
        ("新浪全市场个股", _load_stocks_from_sina_direct),
        ("腾讯沪深A股批量行情", _load_stocks_from_tencent_quotes),
    ]
    errors: list[str] = []
    for label, loader in loaders:
        try:
            frame = loader()
            if not frame.empty:
                return frame, label, ""
        except Exception as exc:
            errors.append(f"{label}失败：{_humanize_market_error(exc)}")
    return _demo_stocks(), "演示数据", "；".join(errors) or "个股失败：未知原因"


def _load_sectors() -> tuple[pd.DataFrame, str, str]:
    """加载板块列表，失败时仅回退板块数据。"""
    try:
        frame = _load_sectors_from_ths()
        return frame, "同花顺行业板块", ""
    except Exception as exc:
        return _demo_sectors(), "演示数据", f"板块失败：{_humanize_market_error(exc)}"


def _load_stocks_from_sina_direct() -> pd.DataFrame:
    """直接从新浪获取全市场个股列表，并强制直连。"""
    from akshare.stock.cons import zh_sina_a_stock_count_url, zh_sina_a_stock_payload, zh_sina_a_stock_url
    from akshare.utils import demjson

    count_response = _direct_get(
        zh_sina_a_stock_count_url,
        timeout=8,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://vip.stock.finance.sina.com.cn/",
        },
    )
    count_response.raise_for_status()
    match = re.findall(r"\d+", count_response.text)
    if not match:
        raise ValueError("新浪个股总页数接口未返回有效内容")

    total_count = int(match[0])
    page_count = total_count // 80 + (1 if total_count % 80 else 0)

    frames: list[pd.DataFrame] = []
    payload = zh_sina_a_stock_payload.copy()
    for page in range(1, page_count + 1):
        payload.update({"page": str(page)})
        response = _direct_get(
            zh_sina_a_stock_url,
            timeout=8,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://vip.stock.finance.sina.com.cn/",
            },
            params=payload,
        )
        response.raise_for_status()
        data_json = demjson.decode(response.text)
        page_frame = pd.DataFrame(data_json)
        if not page_frame.empty:
            frames.append(page_frame)

    if not frames:
        raise ValueError("新浪全市场个股接口未返回有效内容")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.astype(
        {
            "trade": "float",
            "pricechange": "float",
            "changepercent": "float",
            "buy": "float",
            "sell": "float",
            "settlement": "float",
            "open": "float",
            "high": "float",
            "low": "float",
            "volume": "float",
            "amount": "float",
        }
    )
    merged.columns = [
        "代码",
        "_",
        "名称",
        "最新价",
        "涨跌额",
        "涨跌幅",
        "买入",
        "卖出",
        "昨收",
        "今开",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "时间戳",
        "_",
        "_",
        "_",
        "_",
        "_",
    ]
    merged = merged[
        [
            "代码",
            "名称",
            "最新价",
            "涨跌额",
            "涨跌幅",
            "成交量",
            "成交额",
        ]
    ].copy()
    return _normalize(merged, STOCK_COLUMNS)


def _load_stocks_from_tencent_quotes() -> pd.DataFrame:
    """用腾讯批量行情按代码空间探测沪深 A 股，作为非东方财富备选源。"""
    quote_batches = _batched(_candidate_a_share_quote_codes(), 120)
    rows: list[dict[str, object]] = []
    failed_batches = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_load_tencent_quote_batch, batch) for batch in quote_batches]
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except requests.RequestException:
                failed_batches += 1
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("腾讯批量行情接口未返回有效内容")
    if len(frame) < 3000:
        raise ValueError(f"腾讯批量行情有效股票数量过少：{len(frame)}，失败批次：{failed_batches}")
    return _normalize(frame, STOCK_COLUMNS)


def _load_tencent_quote_batch(quote_codes: list[str]) -> list[dict[str, object]]:
    """加载一批腾讯行情代码，并返回解析后的有效股票。"""
    response = _direct_get(
        "https://qt.gtimg.cn/q=" + ",".join(quote_codes),
        timeout=6,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://stockapp.finance.qq.com/",
        },
    )
    response.raise_for_status()
    text = response.content.decode("gbk", errors="ignore")
    return _parse_tencent_quote_lines(text)


def _candidate_a_share_quote_codes() -> list[str]:
    """生成腾讯行情可识别的沪深 A 股候选代码空间。"""
    ranges = [
        ("sz", "000", 1, 1000),
        ("sz", "001", 0, 1000),
        ("sz", "002", 0, 1000),
        ("sz", "003", 0, 1000),
        ("sz", "300", 0, 1000),
        ("sz", "301", 0, 1000),
        ("sh", "600", 0, 1000),
        ("sh", "601", 0, 1000),
        ("sh", "603", 0, 1000),
        ("sh", "605", 0, 1000),
        ("sh", "688", 0, 1000),
    ]
    return [
        f"{market}{prefix}{suffix:03d}"
        for market, prefix, start, stop in ranges
        for suffix in range(start, stop)
    ]


def _parse_tencent_quote_lines(text: str) -> list[dict[str, object]]:
    """解析腾讯 qt.gtimg.cn 批量行情文本。"""
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        if '="' not in line:
            continue
        body = line.split('="', 1)[1].rstrip('";')
        fields = body.split("~")
        if len(fields) < 39 or not fields[2]:
            continue
        rows.append(
            {
                "代码": fields[2],
                "名称": fields[1],
                "最新价": fields[3],
                "涨跌额": fields[31],
                "涨跌幅": fields[32],
                "成交量": fields[36],
                "成交额": fields[37],
                "换手率": fields[38],
                "市盈率-动态": fields[39] if len(fields) > 39 else None,
            }
        )
    return rows


def _batched(items: list[str], size: int) -> list[list[str]]:
    """按固定大小拆分批量行情请求。"""
    return [items[index : index + size] for index in range(0, len(items), size)]


def _load_sectors_from_ths() -> pd.DataFrame:
    """通过同花顺行业板块一览表获取板块强弱数据。"""
    import akshare as ak

    frame = ak.stock_board_industry_summary_ths()
    normalized = _normalize(frame, SECTOR_THS_COLUMNS)
    if normalized.empty:
        raise ValueError("同花顺行业板块接口未返回有效内容")
    return normalized


def _load_sector_history_from_ths_industry(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """通过同花顺行业板块接口获取历史行情。"""
    import akshare as ak

    frame = ak.stock_board_industry_index_ths(symbol=symbol, start_date=start_date, end_date=end_date)
    normalized = _normalize_sector_history(frame)
    if normalized.empty:
        raise ValueError("同花顺行业板块历史接口未返回有效内容")
    return normalized


def _load_sector_history_from_ths_concept(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """通过同花顺概念板块接口获取历史行情。"""
    import akshare as ak

    frame = ak.stock_board_concept_index_ths(symbol=symbol, start_date=start_date, end_date=end_date)
    normalized = _normalize_sector_history(frame)
    if normalized.empty:
        raise ValueError("同花顺概念板块历史接口未返回有效内容")
    return normalized


def _load_major_indices_from_sina() -> pd.DataFrame:
    """直接从新浪行情接口获取三大指数，尽量降低依赖面。"""
    response = _direct_get(
        "https://hq.sinajs.cn/list=s_sh000001,s_sz399001,s_sz399006",
        timeout=8,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn/",
        },
    )
    response.raise_for_status()
    response.encoding = "gbk"

    rows: list[dict[str, object]] = []
    for raw_line in response.text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SINA_INDEX_LINE_PATTERN.match(line)
        if not match:
            continue
        code = match.group("code")
        parts = [item.strip() for item in match.group("body").split(",")]
        if len(parts) < 6:
            continue
        rows.append(
            {
                "code": code.replace("s_sh", "").replace("s_sz", ""),
                "name": parts[0] or SINA_INDEX_CODES.get(code, code),
                "price": parts[1],
                "change": parts[2],
                "change_pct": parts[3],
                "volume": parts[4],
                "turnover": parts[5],
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("新浪三大指数接口未返回有效内容")
    return frame


def _direct_get(
    url: str,
    timeout: int = 8,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    session: requests.Session | None = None,
    retries: int = 2,
) -> requests.Response:
    """发起强制直连请求，绕过系统和终端中的代理设置。"""
    active_session = session or requests.Session()
    should_close = session is None
    try:
        active_session.trust_env = False
        last_error: requests.RequestException | None = None
        for _ in range(retries + 1):
            try:
                request_headers = headers or {}
                return active_session.get(
                    url,
                    params=params,
                    headers=request_headers,
                    timeout=timeout,
                    proxies={"http": None, "https": None},
                )
            except requests.RequestException as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("请求未执行")
    finally:
        if should_close:
            active_session.close()


def _normalize(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """把数据源中可用的列重命名为应用内部字段。"""
    available = {key: value for key, value in mapping.items() if key in frame.columns}
    normalized = frame.rename(columns=available)
    normalized = normalized.loc[:, ~normalized.columns.duplicated()].copy()
    wanted = list(dict.fromkeys(mapping.values()))
    return normalized.reindex(columns=wanted).copy()


def _humanize_market_error(exc: Exception) -> str:
    """把底层网络异常转换成更容易定位的问题描述。"""
    message = str(exc)
    lowered = message.lower()
    if "proxyerror" in lowered or "unable to connect to proxy" in lowered:
        return f"{message}；已识别为代理连接异常，请检查系统代理或网络转发规则。"
    if "remotedisconnected" in lowered or "connection aborted" in lowered:
        return f"{message}；远端连接被提前关闭，通常属于网络链路、网关或安全软件拦截问题。"
    if "name resolution" in lowered or "nodename nor servname provided" in lowered:
        return f"{message}；域名解析失败，请检查 DNS 或网络连通性。"
    return message


def _clean_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """尽可能把非标识列转换成数值类型。"""
    result = frame.copy()
    for column in result.columns:
        if column not in {"code", "name", "leader"}:
            # 无法转换的单元格转成 NaN，图表和表格仍可继续展示。
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _normalize_sector_history(frame: pd.DataFrame) -> pd.DataFrame:
    """统一板块历史行情字段，并补齐趋势图所需指标。"""
    normalized = _normalize(frame, SECTOR_HISTORY_COLUMNS)
    if normalized.empty:
        return normalized

    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    for column in normalized.columns:
        if column != "date":
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    if normalized.empty:
        return normalized

    if "change_pct" not in normalized or normalized["change_pct"].isna().all():
        normalized["change_pct"] = normalized["close"].pct_change().fillna(0).mul(100)
    else:
        normalized["change_pct"] = normalized["change_pct"].fillna(normalized["close"].pct_change().fillna(0).mul(100))

    base_close = normalized["close"].iloc[0]
    if pd.isna(base_close) or base_close == 0:
        normalized["cum_return_pct"] = 0.0
    else:
        normalized["cum_return_pct"] = normalized["close"].div(base_close).sub(1).mul(100)
    return normalized.reset_index(drop=True)


def market_breadth(stocks: pd.DataFrame) -> dict[str, float]:
    """根据个股数据计算一个简洁的市场宽度摘要。"""
    if stocks.empty or "change_pct" not in stocks:
        return {"up": 0, "down": 0, "flat": 0, "limit_up": 0, "limit_down": 0}
    changes = stocks["change_pct"].fillna(0)
    # 这里用近似阈值判断涨停/跌停附近，满足复盘场景即可。
    return {
        "up": int((changes > 0).sum()),
        "down": int((changes < 0).sum()),
        "flat": int((changes == 0).sum()),
        "limit_up": int((changes >= 9.8).sum()),
        "limit_down": int((changes <= -9.8).sum()),
    }


def top_rows(frame: pd.DataFrame, column: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    """按指定指标返回前 N 行或后 N 行，并安全处理缺失字段。"""
    if frame.empty or column not in frame:
        return frame.head(0)
    return frame.sort_values(column, ascending=ascending).head(n)


def _demo_indices() -> pd.DataFrame:
    """在实时数据不可用时使用的小型固定指数样例。"""
    return pd.DataFrame(
        [
            {"code": "000001", "name": "上证指数", "price": 3128.44, "change_pct": 0.72, "change": 22.31, "volume": 322000000, "turnover": 4100},
            {"code": "399001", "name": "深证成指", "price": 9876.20, "change_pct": 1.14, "change": 111.26, "volume": 456000000, "turnover": 5200},
            {"code": "399006", "name": "创业板指", "price": 1888.05, "change_pct": -0.31, "change": -5.86, "volume": 168000000, "turnover": 2100},
        ]
    )


def _demo_sectors() -> pd.DataFrame:
    """用于离线开发和演示的固定板块样例。"""
    rows = [
        ("半导体", 3.42, 28, 6, "中芯国际", 6.8),
        ("机器人", 2.86, 35, 9, "埃斯顿", 7.2),
        ("券商", 1.78, 41, 7, "东方财富", 4.3),
        ("新能源车", 0.64, 52, 31, "宁德时代", 2.2),
        ("银行", -0.22, 12, 29, "招商银行", 0.9),
        ("白酒", -1.18, 7, 24, "贵州茅台", -0.4),
        ("煤炭", -2.31, 4, 32, "中国神华", -1.6),
    ]
    return pd.DataFrame(
        [
            {
                "name": name,
                "price": 1000 + idx * 21,
                "change_pct": change,
                "change": change * 3,
                "market_cap": 8000 - idx * 430,
                "turnover_rate": 1.2 + idx * 0.18,
                "up_count": up,
                "down_count": down,
                "leader": leader,
                "leader_change_pct": leader_change,
            }
            for idx, (name, change, up, down, leader, leader_change) in enumerate(rows)
        ]
    )


def _demo_stocks() -> pd.DataFrame:
    """用于离线测试表格和图表的固定个股样例。"""
    names = ["中芯国际", "东方财富", "宁德时代", "贵州茅台", "比亚迪", "招商银行", "工业富联", "赛力斯", "中国平安", "紫金矿业"]
    changes = [6.8, 4.3, 2.2, -0.4, 1.6, 0.9, 5.1, -3.2, -0.7, 2.9]
    return pd.DataFrame(
        [
            {
                "code": f"60{idx:04d}",
                "name": name,
                "price": round(12 + idx * 8.6, 2),
                "change_pct": changes[idx],
                "change": round(changes[idx] / 100 * (12 + idx * 8.6), 2),
                "volume": 1000000 + idx * 210000,
                "turnover": 120 + idx * 41,
                "turnover_rate": round(0.8 + idx * 0.27, 2),
                "pe_ttm": round(12 + idx * 3.5, 2),
            }
            for idx, name in enumerate(names)
        ]
    )


def _demo_sector_history(name: str, lookback_days: int) -> pd.DataFrame:
    """构造稳定的板块演示走势，保证离线时弹窗也可用。"""
    periods = max(40, min(260, lookback_days))
    dates = pd.bdate_range(end=datetime.now().date(), periods=periods)
    seed = sum(ord(char) for char in name)
    base = 920 + seed % 180
    slope = ((seed % 17) - 8) * 0.42
    amplitude = 12 + seed % 9

    closes: list[float] = []
    for index in range(periods):
        wave = math.sin((index + seed % 13) / 4.8) * amplitude
        pulse = math.cos((index + seed % 7) / 10.5) * (amplitude * 0.45)
        closes.append(round(base + slope * index + wave + pulse, 2))

    rows: list[dict[str, float | pd.Timestamp]] = []
    previous_close = closes[0]
    for index, date in enumerate(dates):
        close = closes[index]
        if index == 0:
            open_price = round(close * (1 - 0.0035), 2)
            change_pct = 0.0
        else:
            open_price = round(previous_close * (1 + math.sin(index / 5.5) * 0.004), 2)
            change_pct = round((close / previous_close - 1) * 100, 2) if previous_close else 0.0
        high = round(max(open_price, close) * 1.011, 2)
        low = round(min(open_price, close) * 0.989, 2)
        rows.append(
            {
                "date": date,
                "open": open_price,
                "close": close,
                "high": high,
                "low": low,
                "change_pct": change_pct,
                "volume": 120000 + index * 380 + seed % 5000,
                "turnover": 1.6e8 + index * 2.3e6 + (seed % 23) * 1e6,
            }
        )
        previous_close = close

    history = pd.DataFrame(rows)
    history["cum_return_pct"] = history["close"].div(history["close"].iloc[0]).sub(1).mul(100)
    return history
