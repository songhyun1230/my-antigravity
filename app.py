import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import ssl
import scipy.stats as stats

# macOS SSL Certificate Error Fix
ssl._create_default_https_context = ssl._create_unverified_context

# ----------------- Configuration & Initialization -----------------
st.set_page_config(page_title="Advanced Quant Backtester", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Institutional-Grade Quant Backtesting Engine")

# ----------------- Helper Functions -----------------
@st.cache_data
def load_kospi_list():
    import urllib.request
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    url = 'https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13&marketType=stockMkt'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req, context=ctx).read()
    df = pd.read_html(html, header=0)[0]
    df = df.rename(columns={'회사명': 'Name', '종목코드': 'Code'})
    df['Code'] = df['Code'].astype(str).str.zfill(6)
    df['YF_Symbol'] = df['Code'] + '.KS'
    return df[['Name', 'YF_Symbol']].dropna()

@st.cache_data
def fetch_data(symbol, start, end):
    # 1. Fetch Target Stock Data
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.copy()

    # 2. Fetch Macro Data (VIX, S&P 500)
    vix = yf.download('^VIX', start=start, end=end, progress=False)
    sp500 = yf.download('^GSPC', start=start, end=end, progress=False)
    
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.droplevel(1)
    if isinstance(sp500.columns, pd.MultiIndex): sp500.columns = sp500.columns.droplevel(1)

    df['VIX'] = vix['Close']
    df['SP500'] = sp500['Close']
    df['SP500_SMA200'] = df['SP500'].rolling(window=200).mean()

    # 3. Fetch US 10Y Treasury (FRED)
    try:
        dgs10 = web.DataReader('DGS10', 'fred', start, end)
        df = df.join(dgs10, how='left')
        df['DGS10'] = df['DGS10'].ffill() # Forward fill missing treasury yields
    except Exception as e:
        df['DGS10'] = 0.0 # Fallback
        
    df = df.dropna(subset=['Close']) # Ensure price exists
    return df

def apply_indicators(df):
    # Strategy A: Volatility Breakout
    df['Range'] = df['High'].shift(1) - df['Low'].shift(1)
    df['Breakout_Target'] = df['Open'] + (df['Range'] * breakout_k)
    
    # Strategy B: MACD & RSI
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Strategy C: Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    return df

# ----------------- Sidebar Variables -----------------
st.sidebar.header("⚙️ Simulation Settings")

kospi_df = load_kospi_list()
stock_dict = dict(zip(kospi_df['Name'], kospi_df['YF_Symbol']))
stock_names = list(stock_dict.keys())
selected_name = st.sidebar.selectbox("Select Asset", stock_names, index=stock_names.index("삼성전자") if "삼성전자" in stock_names else 0)
selected_symbol = stock_dict[selected_name]

# Duration
years_back = st.sidebar.slider("Test Duration (Years)", 1, 10, 3)
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * years_back)

st.sidebar.markdown("---")
st.sidebar.subheader("Constraints")
initial_capital = st.sidebar.number_input("Initial Capital (KRW)", value=10000000, step=1000000)
commission_pct = st.sidebar.number_input("Commission (%)", value=0.015, step=0.005) / 100
slippage_pct = st.sidebar.number_input("Slippage (%)", value=0.05, step=0.01) / 100
position_sizing = st.sidebar.slider("Position Size (% of Capital)", 10, 100, 100) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Selection")
strategy_type = st.sidebar.radio("Active Strategy", [
    "A: Volatility Breakout",
    "B: Momentum (MACD+RSI)",
    "C: Mean Reversion (BB)"
])
if "A: Volatility Breakout" in strategy_type:
    breakout_k = st.sidebar.slider("Breakout Multiplier (K)", 0.1, 1.0, 0.5, 0.1)
else:
    breakout_k = 0.5

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Risk Modules")
use_macro_filter = st.sidebar.checkbox("Apply Macro Regime Filter", value=True, help="Avoid trading if VIX >= 30 or S&P 500 < 200 SMA")
use_kelly = st.sidebar.checkbox("Use Kelly Criterion", value=False, help="Dynamically adjust bet size based on recent 6-month win rate")


# ----------------- Core Backtester Engine -----------------
@st.cache_data
def run_backtest(df_raw, initial_cap, comm_pct, slip_pct, pos_size_fixed, strat_idx, macro_filter, kelly_filter):
    df = df_raw.copy()
    df = apply_indicators(df)
    
    # Initialize Engine Variables
    capital = initial_cap
    cash = capital
    position = 0
    buy_price = 0
    
    equity_curve = []
    buy_markers = []
    sell_markers = []
    
    trade_returns = [] # Track trade R to calculate Kelly
    is_in_trade = False
    
    # Pre-calculate condition masks to optimize loop
    vix_mask = df['VIX'] >= 30
    bear_market_mask = df['SP500'] < df['SP500_SMA200']
    
    macds = df['MACD'].values
    signals = df['Signal_Line'].values
    rsis = df['RSI'].values
    
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    breakout_targets = df['Breakout_Target'].values
    
    bb_lowers = df['BB_Lower'].values
    bb_uppers = df['BB_Upper'].values
    
    dates = df.index
    n = len(df)
    
    for i in range(200, n): # Start after 200 days to allow SMA200 to populate
        # Daily Update Equity
        current_equity = cash + (position * closes[i])
        equity_curve.append(current_equity)
        buy_markers.append(np.nan)
        sell_markers.append(np.nan)
        
        # 1. Macro Regime Check
        regime_blocked = False
        if macro_filter:
            if vix_mask.iloc[i] or bear_market_mask.iloc[i]:
                regime_blocked = True
                
        # If macro environment is bad and we are in a trade, force exit (Cash Position)
        if regime_blocked and is_in_trade:
            # Force Exit
            sell_p = closes[i] * (1 - slip_pct)
            trade_val = position * sell_p
            cash += trade_val - (trade_val * comm_pct)
            trade_ret = (sell_p / buy_price) - 1
            trade_returns.append((dates[i], trade_ret))
            
            position = 0
            is_in_trade = False
            sell_markers[-1] = closes[i]
            continue
            
        # 2. Strategy Logic
        if strat_idx == 'A': # Volatility Breakout
            if not is_in_trade and not regime_blocked:
                # Intraday Breakout (Assuming we caught it during the day)
                if highs[i] > breakout_targets[i] and not pd.isna(breakout_targets[i]):
                    entry_p = breakout_targets[i] * (1 + slip_pct)
                    
                    # Size determination
                    target_exposure = current_equity * pos_size_fixed
                    if kelly_filter and len(trade_returns) > 5:
                        # Calculate recent 6m trades loosely
                        recent = [t[1] for t in trade_returns[-20:]]
                        wins = sum(1 for r in recent if r > 0)
                        win_rate = wins / len(recent)
                        avg_win = np.mean([r for r in recent if r > 0]) if wins > 0 else 0
                        avg_loss = abs(np.mean([r for r in recent if r <= 0])) if (len(recent)-wins) > 0 else 0.0001
                        r_factor = avg_win / avg_loss if avg_loss != 0 else 1
                        kelly_f = win_rate - ((1 - win_rate) / r_factor)
                        kelly_f = max(0.0, min(0.5, kelly_f)) # Cap at 50%
                        target_exposure = current_equity * kelly_f
                        
                    target_exposure = min(target_exposure, cash) # Can't spend more than cash
                    
                    if target_exposure > 0:
                        position = (target_exposure - (target_exposure * comm_pct)) / entry_p
                        cash -= target_exposure
                        buy_price = entry_p
                        is_in_trade = True
                        buy_markers[-1] = entry_p
            
            elif is_in_trade:
                # Sell at next Open
                sell_p = opens[i] * (1 - slip_pct)
                trade_val = position * sell_p
                cash += trade_val - (trade_val * comm_pct)
                trade_ret = (sell_p / buy_price) - 1
                trade_returns.append((dates[i], trade_ret))
                
                position = 0
                is_in_trade = False
                sell_markers[-1] = opens[i]
                
        elif strat_idx == 'B': # MACD + RSI
            if not is_in_trade and not regime_blocked:
                # MACD Croosover UP and RSI > 50
                if macds[i] > signals[i] and macds[i-1] <= signals[i-1] and rsis[i] > 50:
                    entry_p = closes[i] * (1 + slip_pct)
                    target_exposure = current_equity * pos_size_fixed
                    
                    if kelly_filter and len(trade_returns) > 5:
                        recent = [t[1] for t in trade_returns[-20:]]
                        wins = sum(1 for r in recent if r > 0)
                        win_rate = wins / len(recent)
                        avg_win = np.mean([r for r in recent if r > 0]) if wins > 0 else 0
                        avg_loss = abs(np.mean([r for r in recent if r <= 0])) if (len(recent)-wins) > 0 else 0.0001
                        r_factor = avg_win / avg_loss if avg_loss != 0 else 1
                        kelly_f = win_rate - ((1 - win_rate) / r_factor)
                        kelly_f = max(0.0, min(0.5, kelly_f))
                        target_exposure = current_equity * kelly_f
                        
                    target_exposure = min(target_exposure, cash)
                    if target_exposure > 0:
                        position = (target_exposure - (target_exposure * comm_pct)) / entry_p
                        cash -= target_exposure
                        buy_price = entry_p
                        is_in_trade = True
                        buy_markers[-1] = entry_p
            
            elif is_in_trade:
                # Exit MACD crossover down
                if macds[i] < signals[i] and macds[i-1] >= signals[i-1]:
                    sell_p = closes[i] * (1 - slip_pct)
                    trade_val = position * sell_p
                    cash += trade_val - (trade_val * comm_pct)
                    trade_ret = (sell_p / buy_price) - 1
                    trade_returns.append((dates[i], trade_ret))
                    
                    position = 0
                    is_in_trade = False
                    sell_markers[-1] = closes[i]

        elif strat_idx == 'C': # Mean Reversion (Bollinger)
            if not is_in_trade and not regime_blocked:
                # Buy when crossing UP over lower BB
                if closes[i] > bb_lowers[i] and closes[i-1] <= bb_lowers[i-1]:
                    entry_p = closes[i] * (1 + slip_pct)
                    target_exposure = current_equity * pos_size_fixed
                    
                    if kelly_filter and len(trade_returns) > 5:
                        recent = [t[1] for t in trade_returns[-20:]]
                        wins = sum(1 for r in recent if r > 0)
                        win_rate = wins / len(recent)
                        avg_win = np.mean([r for r in recent if r > 0]) if wins > 0 else 0
                        avg_loss = abs(np.mean([r for r in recent if r <= 0])) if (len(recent)-wins) > 0 else 0.0001
                        r_factor = avg_win / avg_loss if avg_loss != 0 else 1
                        kelly_f = win_rate - ((1 - win_rate) / r_factor)
                        kelly_f = max(0.0, min(0.5, kelly_f))
                        target_exposure = current_equity * kelly_f
                        
                    target_exposure = min(target_exposure, cash)
                    if target_exposure > 0:
                        position = (target_exposure - (target_exposure * comm_pct)) / entry_p
                        cash -= target_exposure
                        buy_price = entry_p
                        is_in_trade = True
                        buy_markers[-1] = entry_p
            
            elif is_in_trade:
                # Sell when touching Upper BB
                if highs[i] >= bb_uppers[i]:
                    sell_p = bb_uppers[i] * (1 - slip_pct) # Optimistic limit fill scenario
                    trade_val = position * sell_p
                    cash += trade_val - (trade_val * comm_pct)
                    trade_ret = (sell_p / buy_price) - 1
                    trade_returns.append((dates[i], trade_ret))
                    
                    position = 0
                    is_in_trade = False
                    sell_markers[-1] = bb_uppers[i]

    # Post Processing
    align_index = dates[200:n]
    result_df = pd.DataFrame(index=align_index)
    result_df['Equity'] = equity_curve
    result_df['Buy_Marker'] = buy_markers
    result_df['Sell_Marker'] = sell_markers
    
    # Calculate Buy & Hold Equity 
    # (Buy on day 200 with Initial Cap)
    bnh_shares = initial_cap / closes[200]
    result_df['BnH_Equity'] = bnh_shares * closes[200:n]
    
    return result_df, trade_returns, df

# ----------------- Main Execution -----------------
with st.spinner("Fetching data and running simulations..."):
    df_raw = fetch_data(selected_symbol, start_date, end_date)

if df_raw is None or df_raw.empty:
    st.error("Failed to fetch data for the selected symbol.")
    st.stop()

strat_char = strategy_type.split(":")[0]
result_df, trades, chart_df = run_backtest(df_raw, initial_capital, commission_pct, slippage_pct, position_sizing, strat_char, use_macro_filter, use_kelly)

# ----------------- Metrics Calculation -----------------
final_equity = result_df['Equity'].iloc[-1]
returns = result_df['Equity'].pct_change().dropna()
trading_days = len(returns)
years = trading_days / 252

# CAGR
cagr = ((final_equity / initial_capital) ** (1/years)) - 1 if years > 0 else 0

# MDD
cum_max = result_df['Equity'].cummax()
drawdown = (result_df['Equity'] - cum_max) / cum_max
mdd = drawdown.min()

# Sharpe Ratio (Rf = 2%)
rf_daily = 0.02 / 252
sharpe = (returns.mean() - rf_daily) / returns.std() * np.sqrt(252) if returns.std() != 0 else 0

# Profit Factor
profit_trades = sum(t[1] for t in trades if t[1] > 0)
loss_trades = abs(sum(t[1] for t in trades if t[1] < 0))
profit_factor = profit_trades / loss_trades if loss_trades != 0 else float('inf')

# Value at Risk (VaR 95%)
var_95 = np.percentile(returns, 5)
var_amount = final_equity * abs(var_95)

# --- Top Banner Metrics ---
col1, col2, col3, col4 = st.columns(4)

def stylize_metric(label, value_str, sub_text, color):
    return f"""
    <div style="padding: 15px; border-radius: 10px; background-color: #262730; border-left: 5px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <p style="color: #9aa0a6; margin-bottom: 5px; font-size: 14px; font-weight: 600;">{label}</p>
        <h3 style="margin: 0; font-size: 28px; color: white;">{value_str}</h3>
        <p style="margin: 5px 0 0 0; color: gray; font-size: 13px;">{sub_text}</p>
    </div>
    """

with col1:
    c_color = "#32cd32" if cagr > 0 else "#ff4b4b"
    st.markdown(stylize_metric("CAGR (연평균 수익률)", f"{cagr*100:.2f}%", f"Final Cap: ₩{final_equity:,.0f}", c_color), unsafe_allow_html=True)
with col2:
    st.markdown(stylize_metric("MDD (최대 낙폭)", f"{mdd*100:.2f}%", "Max peak to trough drop", "#ff4b4b"), unsafe_allow_html=True)
with col3:
    s_color = "#32cd32" if sharpe > 1.0 else ("#ffd700" if sharpe > 0 else "#ff4b4b")
    st.markdown(stylize_metric("Sharpe Ratio", f"{sharpe:.2f}", "Risk-adjusted return", s_color), unsafe_allow_html=True)
with col4:
    pf_color = "#32cd32" if profit_factor > 1.0 else "#ff4b4b"
    st.markdown(stylize_metric("Profit Factor", f"{profit_factor:.2f}", f"Total len trades: {len(trades)}", pf_color), unsafe_allow_html=True)

st.write("")
st.warning(f"⚠️ **Risk Analysis (VaR):** Based on the 95% confidence level, the maximum expected daily loss is **₩{var_amount:,.0f}**.")
st.markdown("---")


# ----------------- Visualizations -----------------
st.subheader("📊 Quant Engine Subplots")

# Pre-slice chart_df to match result_df index
chart_df = chart_df.loc[result_df.index]
if strat_char == 'A':
    ind_name = 'Breakout Target'
    ind_series = chart_df['Breakout_Target']
elif strat_char == 'B':
    ind_name = 'MA 20'
    ind_series = chart_df['Close'].rolling(window=20).mean()
else:
    ind_name = 'BB_Lower/Upper'

fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03, subplot_titles=('Price & Trade Entry/Exit', 'Equity Curve Comparison', 'Underwater Drawdown Area'),
                    row_heights=[0.5, 0.3, 0.2])

# 1. Price Chart
fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='Price'), row=1, col=1)

if strat_char == 'C':
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Upper'], line=dict(color='gray', dash='dot'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Lower'], line=dict(color='gray', dash='dot'), name='BB Lower'), row=1, col=1)
elif strat_char == 'A':
    fig.add_trace(go.Scatter(x=chart_df.index, y=ind_series, line=dict(color='orange', width=1), name=ind_name), row=1, col=1)
else:
    fig.add_trace(go.Scatter(x=chart_df.index, y=ind_series, line=dict(color='orange', width=1), name=ind_name), row=1, col=1)

# Markers
fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Buy_Marker'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name='Buy Signal'), row=1, col=1)
fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Sell_Marker'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='Sell Signal'), row=1, col=1)

# 2. Equity Curve
fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Equity'], line=dict(color='#00fa9a', width=2), name='Strategy Equity'), row=2, col=1)
fig.add_trace(go.Scatter(x=result_df.index, y=result_df['BnH_Equity'], line=dict(color='gray', width=1, dash='dash'), name='Buy & Hold Equity'), row=2, col=1)

# 3. Drawdown
fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown*100, fill='tozeroy', fillcolor='rgba(255, 75, 75, 0.3)', line=dict(color='red', width=1), name='Drawdown (%)'), row=3, col=1)

fig.update_layout(height=900, template='plotly_dark', xaxis_rangeslider_visible=False, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# ----------------- Monte Carlo Simulation -----------------
st.markdown("---")
st.subheader("🎲 Monte Carlo Future Distribution Simulation (100 Paths, 252 Days)")

with st.spinner("Generating Monte Carlo paths..."):
    # Generate Multi-path paths
    sim_days = 252
    sim_paths = 100
    mu = returns.mean()
    sigma = returns.std()
    
    # Random normal variables
    Z = np.random.normal(0, 1, (sim_days, sim_paths))
    
    # Daily returns simulation
    drift = mu - (0.5 * sigma**2)
    daily_sim_rets = np.exp(drift + sigma * Z)
    
    # Price paths starting from Final Equity
    price_paths = np.zeros_like(daily_sim_rets)
    price_paths[0] = final_equity
    for t in range(1, sim_days):
        price_paths[t] = price_paths[t-1] * daily_sim_rets[t]
        
    # Plotly Rendering for Monte Carlo
    mc_fig = go.Figure()
    x_future = np.arange(sim_days)
    for i in range(sim_paths):
        mc_fig.add_trace(go.Scatter(x=x_future, y=price_paths[:, i], mode='lines', line=dict(width=1, color='rgba(0, 250, 154, 0.1)'), showlegend=False))
        
    # Add median path
    median_path = np.median(price_paths, axis=1)
    mc_fig.add_trace(go.Scatter(x=x_future, y=median_path, mode='lines', line=dict(color='white', width=3), name='Median Expected Path'))

    mc_fig.update_layout(height=400, template='plotly_dark', xaxis_title="Future Days", yaxis_title="Portfolio Value (KRW)")
    st.plotly_chart(mc_fig, use_container_width=True)

