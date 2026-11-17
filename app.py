# ==================== 强制禁用代理 ====================
import os
for var in ['http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    if var in os.environ:
        try:
            del os.environ[var]
        except:
            pass

# ==================== 导入库 ====================
import streamlit as st
import time
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from openai import OpenAI
import baostock as bs
import akshare as ak

# ==================== 页面配置 ====================
st.set_page_config(page_title="时轮辨龙尺", layout="wide")

# ==================== 安全读取 API Key ====================
try:
    MY_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    st.error("❌ 未找到 API Key，请在 .streamlit/secrets.toml 中配置 DEEPSEEK_API_KEY")
    st.stop()

client = OpenAI(api_key=MY_API_KEY, base_url="https://api.deepseek.com")

# ==================== 历史估值 & 财务指标缓存函数 ====================
@st.cache_data(ttl=600)
def fetch_value_data(symbol):
    try:
        df = ak.stock_value_em(symbol=symbol)
        return df
    except Exception as e:
        st.warning(f"⚠️ 获取历史估值数据失败: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_financial_abstract(symbol):
    try:
        df = ak.stock_financial_abstract(symbol=symbol)
        return df
    except Exception as e:
        st.warning(f"⚠️ 获取财务指标失败: {e}")
        return None

# ==================== baostock 会话管理 ====================
class BaostockSession:
    def __enter__(self):
        lg = bs.login()
        if lg.error_code != '0':
            raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        bs.logout()

def _bs_code(code: str) -> str:
    if code.startswith(('600', '601', '603', '605', '688')):
        return f"sh.{code}"
    elif code.startswith(('4', '8')):
        return f"bj.{code}"
    return f"sz.{code}"

# ==================== 辅助函数 ====================
def get_stock_info_baostock(code: str, session: BaostockSession):
    try:
        rs = bs.query_stock_basic(code=_bs_code(code))
        if rs.error_code != '0':
            return None
        rows = []
        while rs.next():
            row = rs.get_row_data()
            if row:
                rows.append(row)
        return rows[0][1] if rows else None
    except Exception:
        return None

def get_technical_baostock(code: str, session: BaostockSession, days: int = 60) -> dict:
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rs = bs.query_history_k_data_plus(
            code=_bs_code(code),
            fields="date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"
        )
        if rs.error_code != '0':
            return {}
        data = []
        while rs.next():
            row = rs.get_row_data()
            if row:
                data.append(row)
        if not data or len(data) < 20:
            return {}
        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        df = df.sort_values('date')
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['amplitude'] = ((df['high'] - df['low']) / df['close'].shift(1) * 100).fillna(0)

        latest = df.iloc[-1]
        last5 = df.tail(5)
        avg_volume_5 = last5['volume'].mean()
        avg_amount_5 = avg_volume_5 * latest['close'] / 10000
        avg_amplitude_5 = last5['amplitude'].mean()
        today_amount = latest['volume'] * latest['close'] / 10000
        today_amplitude = latest['amplitude']
        vol_ratio = latest['volume'] / avg_volume_5 if avg_volume_5 > 0 else 1.0

        if len(df) >= 6:
            close_5d_ago = df.iloc[-6]['close']
            change_5d = ((latest['close'] - close_5d_ago) / close_5d_ago * 100) if close_5d_ago > 0 else 0
        else:
            change_5d = 0

        return {
            'latest_price': latest['close'],
            'ma5': latest['MA5'],
            'ma20': latest['MA20'],
            'avg_amount_5_yi': avg_amount_5 / 1e4,
            'avg_amplitude_5': avg_amplitude_5,
            'today_amplitude': today_amplitude,
            'today_amount_yi': today_amount / 1e4,
            'vol_ratio_vs_5': vol_ratio,
            'change_5d': change_5d,
        }
    except Exception:
        return {}

def get_stock_news(code: str) -> list:
    if code.startswith(('600', '601', '603', '605', '688')):
        market = 1
    elif code.startswith(('4', '8')):
        market = 2
    else:
        market = 0
    url = (
        f"https://np-listapi.eastmoney.com/comm/wap/getListInfo"
        f"?client=wap&type=1&mTypeAndCode={market}.{code}"
        f"&pageSize=10&pageIndex=1&_={int(time.time() * 1000)}"
    )
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://wap.eastmoney.com/"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = data.get('data', {}).get('list', [])
        results = []
        for item in items:
            title = item.get('Art_Title', '').strip()
            media = item.get('Art_MediaName', '').strip()
            show_time = item.get('Art_ShowTime', '').strip()
            time_part = show_time[11:16] if len(show_time) >= 16 else ''
            if title:
                results.append({'title': title, 'media': media, 'time': time_part})
        return results[:10]
    except Exception:
        return []

def get_price_sina(code: str) -> dict:
    if code.startswith(('600', '601', '603', '605', '688')):
        prefix = "sh"
    elif code.startswith(('4', '8')):
        prefix = "bj"
    else:
        prefix = "sz"
    url = f"https://hq.sinajs.cn/list={prefix}{code}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.encoding = 'gbk'
        text = resp.text
        if '="' in text:
            parts = text.split('="')[1].split('"')[0].split(',')
            if len(parts) >= 10:
                latest = float(parts[3])
                yesterday_close = float(parts[2])
                change_pct = ((latest - yesterday_close) / yesterday_close * 100) if yesterday_close else None
                return {
                    'price': latest,
                    'change_pct': round(change_pct, 2) if change_pct is not None else None,
                    'open': float(parts[1]),
                    'high': float(parts[4]),
                    'low': float(parts[5]),
                    'yesterday_close': yesterday_close,
                    'volume': int(float(parts[6])) if parts[6] else 0,
                    'amount': float(parts[7]) if parts[7] else 0.0,
                    'amount_yi': float(parts[7]) / 1e8 if parts[7] else 0.0,
                }
    except Exception:
        pass
    return None

def get_price_tencent(code: str) -> dict:
    if code.startswith(('600', '601', '603', '605', '688')):
        prefix = "sh"
    elif code.startswith(('4', '8')):
        prefix = "bj"
    else:
        prefix = "sz"
    url = f"https://qt.gtimg.cn/q={prefix}{code}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.encoding = 'gbk'
        text = resp.text
        if '~' not in text:
            return None
        parts = text.split('="')[1].split('"')[0].split('~')
        if len(parts) < 40:
            return None
        latest = float(parts[3])
        yesterday_close = float(parts[4])
        open_price = float(parts[5])
        volume = int(float(parts[6])) * 100
        high = float(parts[33])
        low = float(parts[34])
        amount = float(parts[37]) * 1e4
        change_pct = ((latest - yesterday_close) / yesterday_close * 100) if yesterday_close else None
        return {
            'price': latest,
            'change_pct': round(change_pct, 2) if change_pct is not None else None,
            'open': open_price,
            'high': high,
            'low': low,
            'yesterday_close': yesterday_close,
            'volume': volume,
            'amount': amount,
            'amount_yi': amount / 1e8,
        }
    except (ValueError, IndexError, requests.RequestException):
        return None

def get_price_akshare_single(code: str) -> dict:
    try:
        df = ak.stock_bid_ask_em(symbol=code)
        info = dict(zip(df['item'], df['value']))
        latest = float(info.get('最新', 0))
        yesterday_close = float(info.get('昨收', 0))
        if not latest:
            return None
        change_pct = ((latest - yesterday_close) / yesterday_close * 100) if yesterday_close else None
        return {
            'price': latest,
            'change_pct': round(change_pct, 2) if change_pct is not None else None,
            'open': float(info.get('今开', 0)) or None,
            'high': float(info.get('最高', 0)) or None,
            'low': float(info.get('最低', 0)) or None,
            'yesterday_close': yesterday_close or None,
            'volume': int(float(info.get('总手', 0))) or None,
            'amount': None,
            'amount_yi': None,
        }
    except Exception:
        return None

def get_price_eastmoney_direct(code: str) -> dict:
    if code.startswith(('600', '601', '603', '605', '688')):
        market = 1
    elif code.startswith(('4', '8')):
        market = 2
    else:
        market = 0
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={market}.{code}&fields=f43,f44,f45,f46,f47,f48,f60,f170"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        data = resp.json().get('data')
        if not data or not data.get('f43'):
            return None
        return {
            'price': data['f43'] / 100,
            'change_pct': data.get('f170'),
            'open': data['f46'] / 100,
            'high': data['f44'] / 100,
            'low': data['f45'] / 100,
            'yesterday_close': data['f60'] / 100,
            'volume': data['f47'] * 100,
            'amount': data['f48'],
            'amount_yi': data['f48'] / 1e8,
        }
    except (KeyError, TypeError, ZeroDivisionError, requests.RequestException):
        return None

def get_realtime_price(code: str) -> dict:
    result = get_price_tencent(code)
    if result and result.get('price') is not None:
        return result
    result = get_price_sina(code)
    if result and result.get('price') is not None:
        return result
    result = get_price_akshare_single(code)
    if result and result.get('price') is not None:
        return result
    result = get_price_eastmoney_direct(code)
    if result and result.get('price') is not None:
        return result
    try:
        with BaostockSession() as session:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(
                code=_bs_code(code),
                fields="date,close,volume",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"
            )
            data = []
            while rs.next():
                row = rs.get_row_data()
                if row:
                    data.append(row)
            if data:
                last = data[-1]
                price = float(last[1])
                vol = int(float(last[2])) if last[2] else None
                return {
                    'price': price,
                    'change_pct': None,
                    'open': None,
                    'high': None,
                    'low': None,
                    'yesterday_close': None,
                    'volume': vol,
                    'amount': None,
                    'amount_yi': None,
                }
    except Exception:
        pass
    return {}

def get_fundamentals_akshare(code: str) -> dict:
    result = {}
    try:
        df = ak.stock_a_lg_indicator(symbol=code)
        if not df.empty:
            latest = df.iloc[-1]
            recent_4q = df.tail(4)
            result = {
                'pe_ttm': latest.get('peTTM', None),
                'pb': latest.get('pb', None),
                'roe': latest.get('roe', None),
                'gross_margin': latest.get('grossProfitMargin', None),
                'net_margin': latest.get('netProfitMargin', None),
            }
            if len(recent_4q) >= 4 and 'revenue' in df.columns and 'netProfit' in df.columns:
                revenue_4q = df['revenue'].tail(4).sum()
                profit_4q = df['netProfit'].tail(4).sum()
                if len(df) >= 8:
                    revenue_prev_4q = df['revenue'].iloc[-8:-4].sum()
                    profit_prev_4q = df['netProfit'].iloc[-8:-4].sum()
                    if revenue_prev_4q > 0:
                        result['revenue_yoy'] = (revenue_4q - revenue_prev_4q) / revenue_prev_4q * 100
                    if profit_prev_4q != 0:
                        result['profit_yoy'] = (profit_4q - profit_prev_4q) / abs(profit_prev_4q) * 100
                result['revenue_4q'] = revenue_4q / 1e8
                result['profit_4q'] = profit_4q / 1e8
    except Exception:
        pass

    try:
        forecast_df = ak.stock_yjyg_em(date="2026-06-30")
        if not forecast_df.empty:
            row = forecast_df[forecast_df['代码'] == code]
            if not row.empty:
                result['forecast_type'] = row.iloc[0].get('预告类型', '')
                result['forecast_change'] = row.iloc[0].get('变动幅度', '')
    except Exception:
        pass

    if not result.get('pe_ttm'):
        try:
            with BaostockSession() as session:
                rs = bs.query_history_k_data_plus(
                    code=_bs_code(code),
                    fields="date,close,peTTM,pbMRQ",
                    start_date=(datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                    end_date=datetime.now().strftime("%Y-%m-%d"),
                    frequency="d",
                    adjustflag="2"
                )
                data = []
                while rs.next():
                    row = rs.get_row_data()
                    if row:
                        data.append(row)
                if data:
                    last = data[-1]
                    result['pe_ttm'] = float(last[2]) if len(last) > 2 and last[2] else result.get('pe_ttm')
                    result['pb'] = float(last[3]) if len(last) > 3 and last[3] else result.get('pb')
        except Exception:
            pass
    return result

# ==================== 构建 AI Prompt ====================
def build_diagnosis_prompt(
    stock_code: str,
    stock_name: str,
    price_info: dict,
    tech: dict,
    fundamentals: dict,
    news_items: list,
    value_df: pd.DataFrame = None,
    fin_df: pd.DataFrame = None
) -> tuple:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def safe(v, fmt=".2f"):
        if v is None or pd.isna(v):
            return "N/A"
        try:
            return f"{float(v):{fmt}}"
        except:
            return str(v)

    # ---- 提取历史估值最新数据 ----
    latest_value = {}
    recent_pe = []
    recent_pb = []
    if value_df is not None and not value_df.empty:
        value_df_sorted = value_df.sort_values('数据日期', ascending=False)
        if not value_df_sorted.empty:
            latest_value = value_df_sorted.iloc[0].to_dict()
            if '市盈率-动态' in value_df_sorted.columns:
                recent_pe = value_df_sorted['市盈率-动态'].head(5).tolist()
            if '市净率' in value_df_sorted.columns:
                recent_pb = value_df_sorted['市净率'].head(5).tolist()

    # ---- 提取财务指标最新报告期数据 ----
        fin_latest = {}
    fin_recent = pd.DataFrame()
    if fin_df is not None and not fin_df.empty:
        try:
            indicator_col = None
            for col in fin_df.columns:
                if '指标' in col or '名称' in col:
                    indicator_col = col
                    break
            if indicator_col is None:
                indicator_col = fin_df.columns[0]
            fin_tmp = fin_df.set_index(indicator_col).T
            fin_tmp.index = fin_tmp.index.astype(str)
            fin_tmp.index = pd.to_datetime(fin_tmp.index, format='%Y%m%d', errors='coerce')
            fin_tmp = fin_tmp.dropna(how='all', axis=0)
            if not fin_tmp.empty:
                fin_tmp = fin_tmp.sort_index()
                latest_period = fin_tmp.index[-1]
                fin_latest = fin_tmp.loc[latest_period].to_dict()
                
                # ========== 修改：固定取 2022-2025年报 + 2026最新 ==========
                target_years = [2022, 2023, 2024, 2025]
                selected_dates = []
                
                for year in target_years:
                    mask = (fin_tmp.index.year == year) & (fin_tmp.index.month == 12) & (fin_tmp.index.day == 31)
                    year_end_data = fin_tmp[mask]
                    if not year_end_data.empty:
                        selected_dates.append(year_end_data.index[-1])
                
                mask_2026 = fin_tmp.index.year == 2026
                if mask_2026.any():
                    latest_2026 = fin_tmp[mask_2026].index[-1]
                    selected_dates.append(latest_2026)
                
                selected_dates = sorted(set(selected_dates))
                
                if len(selected_dates) < 2:
                    recent_periods = fin_tmp.index[-4:] if len(fin_tmp) >= 4 else fin_tmp.index
                else:
                    recent_periods = selected_dates
                # ========== 修改结束 ==========
                
                fin_recent = fin_tmp.loc[recent_periods]
        except Exception:
            pass

    data_summary = f"""
## 股票数据

**股票名称**：{stock_name}
**股票代码**：{stock_code}
**诊断时间**：{now_str}

### 一、实时行情
- 最新价：{safe(price_info.get('price'))} 元
- 涨跌幅：{price_info.get('change_pct', 0):+.2f}%
- 开盘价：{safe(price_info.get('open'))} 元
- 最高价：{safe(price_info.get('high'))} 元
- 最低价：{safe(price_info.get('low'))} 元
- 昨日收盘：{safe(price_info.get('yesterday_close'))} 元
- 今日成交额：{safe(price_info.get('amount_yi'))} 亿元
- 今日振幅：{safe(tech.get('today_amplitude'))}%

### 二、技术指标
- 5日均线：{safe(tech.get('ma5'))} 元
- 20日均线：{safe(tech.get('ma20'))} 元
- 股价与5日均线关系：{"站上5日线" if price_info.get('price') and tech.get('ma5') and price_info['price'] > tech['ma5'] else "跌破5日线" if price_info.get('price') and tech.get('ma5') else "数据不足"}
- 股价与20日均线关系：{"站上20日线" if price_info.get('price') and tech.get('ma20') and price_info['price'] > tech['ma20'] else "跌破20日线" if price_info.get('price') and tech.get('ma20') else "数据不足"}
- 量比（较5日均量）：{safe(tech.get('vol_ratio_vs_5'))} 倍
- 近5日日均成交额：{safe(tech.get('avg_amount_5_yi'))} 亿元
- 近5日涨跌幅：{safe(tech.get('change_5d'))}%
- 近5日日均振幅：{safe(tech.get('avg_amplitude_5'))}%

### 三、基本面数据（近期财务表现）
- 最新PE（TTM）：{safe(fundamentals.get('pe_ttm'))} 倍
- 最新市净率（PB）：{safe(fundamentals.get('pb'))} 倍
- ROE：{safe(fundamentals.get('roe'))}%
- 毛利率：{safe(fundamentals.get('gross_margin'))}%
- 净利率：{safe(fundamentals.get('net_margin'))}%
- 近4个季度营收合计：{safe(fundamentals.get('revenue_4q'))} 亿元
- 近4个季度净利润合计：{safe(fundamentals.get('profit_4q'))} 亿元
- 营收同比增速：{safe(fundamentals.get('revenue_yoy'))}%
- 净利润同比增速：{safe(fundamentals.get('profit_yoy'))}%

### 四、历史估值数据（来自东方财富）
- 最新日期：{latest_value.get('数据日期', 'N/A')}
- 最新市盈率（动态）：{safe(latest_value.get('市盈率-动态'))} 倍
- 最新市净率：{safe(latest_value.get('市净率'))} 倍
- 最新总市值：{safe(latest_value.get('总市值'))} 亿元
- 最新流通市值：{safe(latest_value.get('流通市值'))} 亿元
- 近5期PE趋势：{', '.join([safe(p) for p in recent_pe]) if recent_pe else 'N/A'}
- 近5期PB趋势：{', '.join([safe(p) for p in recent_pb]) if recent_pb else 'N/A'}

### 五、财务关键指标（来自新浪财经，最近报告期）
{json.dumps(fin_latest, ensure_ascii=False, indent=2) if fin_latest else '暂无数据'}

**以下为财务关键指标数据（包含2022-2025年年报及2026年最新一期，请全部用于生成表格，不要遗漏任何报告期）**：
{fin_recent.to_string(max_cols=None, max_rows=None) if not fin_recent.empty else '暂无历史财务数据'}

### 六、业绩预告
- 预告类型：{fundamentals.get('forecast_type', '暂无')}
- 变动幅度：{fundamentals.get('forecast_change', '暂无')}

### 七、近期个股新闻
{json.dumps([item['title'] for item in news_items[:5]], ensure_ascii=False, indent=2)}
"""

    system_prompt = f"""你是一位专业的A股市场分析师。请基于用户提供的股票数据，撰写一份完整的个股诊断报告。

**报告日期**：{now_str.split()[0]}

报告必须严格按照以下格式输出，结论由你根据数据自行判断。每条诊断要分成多小条，每个小条都要有具体的分析和结论。**所有表格中的数值必须从提供的数据中提取，不得编造；若数据缺失，请标注“暂无”**。

---

## {{股票名称}}（{{股票代码}}）—— 核心结论

> **免责声明**：以下内容仅为基于公开信息的客观分析，不构成任何投资建议。股市有风险，投资需谨慎，请独立判断。

### [结论标签] 核心结论：[一句话概括]

| 维度 | 判断 |
|------|------|
| **当前适合操作** | [买入/持有/减仓/清仓/观望] |
| **是否适合长线** | [合适/不合适/尚待验证]，[简要理由] |
| **是否适合短线** | [可参与/空间有限/不建议]，[简要理由] |
| **当前状态** | [一句话描述] |
| **综合评级** | [🏆长线布局 / ⏳长线观望 / 🎲短线博弈 / 🗑️直接回避] |

**三个关键词**：[关键词1、关键词2、关键词3]

### 📊 快速画像

| 指标 | 数据 |
|------|------|
| 股价 | [从实时行情提取] |
| 总市值 | [从历史估值最新数据提取，若没有则从基本面或行情估算] |
| 动态市盈率（TTM） | [从最新PE提取] |
| 市净率 | [从最新PB提取] |
| 每股净资产 | [若数据中有则填] |

# 📋 六条核心诊断

### 一、【公司底色】

[基于公司主营业务、行业地位、商业模式进行分析，结合基本面数据中的营收规模、毛利率等]

**底色评级**：[白马/灰马/黑马/野马]

### 二、【财务真相】

[基于下方提供的“财务关键指标数据”，根据实际包含的报告期数量，动态生成表格列。每个报告期作为一列，按时间从左到右排列。]

| 指标 | [报告期1] | [报告期2] | [报告期3] | ...（按实际数据列数自动扩展） |
|------|-----------|-----------|-----------|-----------|
| 营收（亿元） | [数据] | [数据] | [数据] | [数据] |
| 归母净利润（亿元） | [数据] | [数据] | [数据] | [数据] |
| 扣非净利润（亿元） | [若有] | [若有] | [若有] | [若有] |
| 净利率（%） | [数据] | [数据] | [数据] | [数据] |

[结合数据趋势，分析营收增长性、利润质量、盈利能力变化等]

### 三、【估值分析】

[基于提供的PE、PB历史趋势（近5期）以及最新估值水平，与行业平均对比（若缺少行业平均则与自身历史对比），分析当前估值高低，判断是否被低估或高估，并结合成长性给出PEG等分析]

### 四、【情绪与风险】

[基于量价、新闻、公告等分析市场情绪和风险点，包括换手率、振幅、新闻舆情等]

### 五、【治理与风险】

[基于实控人、质押、商誉、现金流等分析，若数据缺失则说明无法评估]

### 六、【操作建议】

| 投资者类型 | 建议 |
|----------|------|
| **未持仓者** | [建议] |
| **轻仓持有者** | [建议] |
| **重仓持有者** | [建议] |
| **想博短线者** | [建议] |

## 🚨 风险提示

[基于数据分析得出的主要风险]

---

*报告生成时间：{now_str}｜数据基于公开财报、公司公告、交易所披露数据*
"""
    return system_prompt, data_summary

def run_diagnosis(
    stock_code: str,
    stock_name: str,
    price_info: dict,
    tech: dict,
    fundamentals: dict,
    news_items: list,
    value_df: pd.DataFrame = None,
    fin_df: pd.DataFrame = None
) -> str:
    system_prompt, user_data = build_diagnosis_prompt(
        stock_code, stock_name, price_info, tech, fundamentals, news_items,
        value_df, fin_df
    )
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_data}
            ],
            timeout=90
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI 分析失败：{str(e)}"

# ==================== 界面 ====================
st.title("📊 时轮辨龙尺")
st.caption("输入股票代码，AI 自动获取数据并生成诊断报告")

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📌 输入个股")
    stock_code = st.text_input(
        "请输入股票代码（6位数字）",
        value="601872",
        help="例如: 000001, 600519, 601872"
    )
    analyze_btn = st.button("🚀 开始诊断", type="primary", use_container_width=True)
    st.info("""
    **系统说明：**
    - 数据源：腾讯财经 + 新浪财经 + 东方财富 + AkShare + baostock
    - 分析引擎：DeepSeek AI
    - 本系统仅供学习研究，不构成投资建议
    """)

if analyze_btn:
    if not stock_code.isdigit() or len(stock_code) != 6:
        st.error("❌ 请输入正确的6位股票代码（例如 601872）")
        st.stop()

    with col2:
        progress_area = st.empty()
        progress_bar = st.progress(0)

    try:
        progress_area.markdown("🔍 **步骤1/6**：正在识别股票信息...")
        progress_bar.progress(10)

        stock_name = stock_code
        with BaostockSession() as session:
            name = get_stock_info_baostock(stock_code, session)
            if name:
                stock_name = name
                progress_area.markdown(f"✅ **步骤1/6**：{stock_name} ({stock_code})")
            else:
                progress_area.markdown(f"⚠️ **步骤1/6**：名称获取失败")

        progress_area.markdown("💰 **步骤2/6**：正在获取实时行情...")
        progress_bar.progress(20)
        price_info = get_realtime_price(stock_code)
        price_text = (
            f"{price_info['price']} 元 ({price_info['change_pct']:+.2f}%)"
            if price_info and price_info.get('price') is not None else "获取失败"
        )
        progress_bar.progress(30)
        progress_area.markdown(f"✅ **步骤2/6**：当前股价 {price_text}")

        progress_area.markdown("📈 **步骤3/6**：正在计算技术指标...")
        progress_bar.progress(40)
        with BaostockSession() as session:
            tech = get_technical_baostock(stock_code, session)
        progress_bar.progress(50)
        progress_area.markdown(f"✅ **步骤3/6**：技术指标计算完成")

        progress_area.markdown("📊 **步骤4/6**：正在获取基本面数据...")
        progress_bar.progress(55)
        fundamentals = get_fundamentals_akshare(stock_code)
        progress_bar.progress(65)

        progress_area.markdown("📉 **步骤5/6**：正在获取历史估值与财务指标...")
        progress_bar.progress(70)
        value_df = fetch_value_data(stock_code)
        fin_df = fetch_financial_abstract(stock_code)
        progress_bar.progress(80)
        progress_area.markdown("✅ **步骤5/6**：历史估值与财务指标获取完成")

        progress_area.markdown("📰 **步骤6/6**：正在获取个股新闻...")
        progress_bar.progress(85)
        news_items = get_stock_news(stock_code)
        progress_bar.progress(90)
        progress_area.markdown(f"✅ **步骤6/6**：新闻 {len(news_items)} 条")

        progress_area.markdown("🧠 **AI 诊断中...**")
        progress_bar.progress(95)

        report = run_diagnosis(
            stock_code, stock_name, price_info, tech, fundamentals, news_items,
            value_df, fin_df
        )

        progress_bar.progress(100)
        progress_area.empty()
        progress_bar.empty()

        st.subheader(f"📊 {stock_name}（{stock_code}）诊断报告")
        st.write(f"**⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**  &nbsp;&nbsp; **💰 {price_text}**")
        st.success("✅ 诊断完成")

        # ==================== 历史估值数据：仅显示最新一日 ====================
        with st.expander("📈 历史估值数据（PE/PB/市值等）", expanded=False):
            if value_df is not None and not value_df.empty:
                value_df_sorted = value_df.sort_values('数据日期', ascending=False)
                latest_value = value_df_sorted.head(1)
                st.dataframe(latest_value, use_container_width=True, height=100)
                st.caption(f"最新数据日期: {latest_value['数据日期'].iloc[0]} ｜ 总记录数: {len(value_df)} 条")
                csv_value = value_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 下载估值历史CSV", csv_value, f"{stock_code}_估值历史.csv", "text/csv")
            else:
                st.warning("未获取到历史估值数据")

        # ==================== 财务关键指标：年报（2022-2025）+ 2026最新一期 ====================
        with st.expander("📊 财务关键指标（新浪财经）", expanded=False):
            if fin_df is not None and not fin_df.empty:
                try:
                    # 1. 识别指标列
                    indicator_col = None
                    for col in fin_df.columns:
                        if '指标' in col or '名称' in col:
                            indicator_col = col
                            break
                    if indicator_col is None:
                        indicator_col = fin_df.columns[0]

                    # 2. 转置，行索引为报告期（如 '20260331'）
                    fin_tmp = fin_df.set_index(indicator_col).T
                    fin_tmp.index = fin_tmp.index.astype(str)
                    fin_tmp.index = pd.to_datetime(fin_tmp.index, format='%Y%m%d', errors='coerce')
                    fin_tmp = fin_tmp.dropna(how='all', axis=0)
                    if fin_tmp.empty:
                        st.warning("⚠️ 无法解析报告期日期，显示原始数据（前10列）")
                        st.dataframe(fin_df.iloc[:, :10], use_container_width=True, height=400)
                    else:
                        # 3. 筛选 >=2022
                        fin_tmp['年份'] = fin_tmp.index.year
                        fin_tmp = fin_tmp[fin_tmp['年份'] >= 2022]
                        if fin_tmp.empty:
                            st.warning("⚠️ 未找到2022年及以后的数据，显示原始数据（前10列）")
                            st.dataframe(fin_df.iloc[:, :10], use_container_width=True, height=400)
                        else:
                            # 4. 按年份分组，每年只取一条
                            annual_series = {}
                            for year, group in fin_tmp.groupby('年份'):
                                if year == 2026:
                                    # 2026年取最新一期（按日期排序取最后一条）
                                    selected = group.sort_index().iloc[-1]
                                else:
                                    # 其他年份取年报（12月31日）
                                    annual_mask = (group.index.month == 12) & (group.index.day == 31)
                                    annual_data = group[annual_mask]
                                    if not annual_data.empty:
                                        selected = annual_data.sort_index().iloc[-1]
                                    else:
                                        # 若无年报则跳过该年
                                        continue
                                annual_series[year] = selected
                            if not annual_series:
                                st.warning("⚠️ 未找到2022-2025年年报及2026年最新数据，显示季度数据（前10列）")
                                st.dataframe(fin_tmp.head(10), use_container_width=True, height=400)
                            else:
                                # 5. 组装成DataFrame（年份为列，指标为行）
                                annual_df = pd.DataFrame(annual_series).T   # 转置后年份为行
                                annual_df = annual_df.sort_index()          # 按年份排序
                                st.dataframe(annual_df.T, use_container_width=True, height=400)  # 再转置使年份为列
                                st.caption(f"数据覆盖年份：{', '.join(map(str, annual_df.index))}（2026年为最新一期季报）")
                    # 下载原始数据
                    csv_fin = fin_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 下载原始财务指标CSV", csv_fin, f"{stock_code}_财务指标.csv", "text/csv")

                except Exception as e:
                    st.warning(f"⚠️ 财务数据处理异常，显示原始数据：{e}")
                    st.dataframe(fin_df.head(10), use_container_width=True, height=400)
                    csv_fin = fin_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 下载原始财务指标CSV", csv_fin, f"{stock_code}_财务指标.csv", "text/csv")
            else:
                st.warning("未获取到财务指标数据")

        # 显示AI报告
        st.markdown(report)
        st.warning("⚠️ 本报告基于公开数据生成，仅供参考，不构成投资建议。")

    except RuntimeError as e:
        st.error(f"❌ 数据连接失败：{e}")
    except Exception as e:
        st.error(f"❌ 分析异常：{e}")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Powered by baostock + AkShare + 东方财富 + DeepSeek</p>",
    unsafe_allow_html=True
)
