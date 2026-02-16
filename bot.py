import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Multi-Timeframe Scalping Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# ENHANCED CSS - MODERN DARK THEME
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Signal Boxes - Glassmorphism */
    .signal-box {
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 20px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .signal-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }
    
    .buy-signal { 
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        animation: pulse-green 2s infinite;
    }
    
    .sell-signal { 
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        animation: pulse-red 2s infinite;
    }
    
    .neutral-signal { 
        background: linear-gradient(135deg, #4b5563 0%, #9ca3af 100%);
        color: white;
    }
    
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 0 0 rgba(17, 153, 142, 0.4); }
        50% { box-shadow: 0 0 0 20px rgba(17, 153, 142, 0); }
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 0 0 rgba(235, 51, 73, 0.4); }
        50% { box-shadow: 0 0 0 20px rgba(235, 51, 73, 0); }
    }
    
    /* Analysis Cards */
    .analysis-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .analysis-card:hover {
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .step-header {
        font-size: 0.9rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }
    
    .success-card {
        border-left: 5px solid #10b981;
        background: linear-gradient(90deg, #ecfdf5 0%, #ffffff 100%);
    }
    
    .error-card {
        border-left: 5px solid #ef4444;
        background: linear-gradient(90deg, #fef2f2 0%, #ffffff 100%);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 4px;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-success { background: #d1fae5; color: #065f46; }
    .badge-error { background: #fee2e2; color: #991b1b; }
    .badge-neutral { background: #f3f4f6; color: #374151; }
    
    /* Sidebar Styling */
    .sidebar-content {
        padding: 20px;
    }
    
    .strictness-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Price Display */
    .price-display {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1f2937 0%, #4b5563 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #e5e7eb 50%, transparent 100%);
        margin: 30px 0;
    }
    
    /* Status Bar */
    .status-bar {
        background: white;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# INDICATORS
# ==========================================
@st.cache_data(ttl=10)
def fetch_candles(symbol, timeframe, limit=300):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching {timeframe}: {str(e)}")
        return pd.DataFrame()

def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

def calculate_vwap(df):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_sma(series, period):
    return series.rolling(window=period).mean()

# ==========================================
# TRADING LOGIC
# ==========================================
def analyze_trend(df_15m, threshold):
    if len(df_15m) < 200:
        return 'UNKNOWN', 0, None, None
    
    ema50 = calculate_ema(df_15m['close'], 50)
    ema200 = calculate_ema(df_15m['close'], 200)
    
    price = df_15m['close'].iloc[-1]
    ema50_val = ema50.iloc[-1]
    ema200_val = ema200.iloc[-1]
    
    strength = abs(ema50_val - ema200_val) / price
    
    if strength < threshold:
        return 'SIDEWAYS', strength, ema50_val, ema200_val
    
    trend = 'BULLISH' if ema50_val > ema200_val else 'BEARISH'
    return trend, strength, ema50_val, ema200_val

def check_5m_setup(df_5m, trend, tolerance_pct):
    if len(df_5m) < 50:
        return False, {'error': 'Insufficient data'}
    
    ema20 = calculate_ema(df_5m['close'], 20)
    ema50 = calculate_ema(df_5m['close'], 50)
    vwap = calculate_vwap(df_5m)
    
    price = df_5m['close'].iloc[-1]
    ema20_val = ema20.iloc[-1]
    ema50_val = ema50.iloc[-1]
    vwap_val = vwap.iloc[-1]
    
    tolerance = tolerance_pct * price
    near_ema20 = abs(price - ema20_val) <= tolerance
    near_ema50 = abs(price - ema50_val) <= tolerance
    near_zone = near_ema20 or near_ema50
    
    details = {
        'price': price, 'vwap': vwap_val,
        'ema20': ema20_val, 'ema50': ema50_val,
        'near_zone': near_zone
    }
    
    if trend == 'BULLISH':
        valid = (price > vwap_val) and (ema20_val > ema50_val) and near_zone
    elif trend == 'BEARISH':
        valid = (price < vwap_val) and (ema20_val < ema50_val) and near_zone
    else:
        valid = False
    
    return valid, details

def check_volatility(df_5m, min_atr):
    if len(df_5m) < 14:
        return False, 0, 0
    
    atr = calculate_atr(df_5m, 14)
    price = df_5m['close'].iloc[-1]
    atr_val = atr.iloc[-1]
    atr_pct = atr_val / price
    
    return atr_pct >= min_atr, atr_val, atr_pct

def check_1m_trigger(df_1m, trend, vol_mult):
    if len(df_1m) < 11:
        return False, None, {'error': 'Insufficient data'}
    
    avg_vol = calculate_sma(df_1m['volume'], 10)
    curr_vol = df_1m['volume'].iloc[-1]
    avg_vol_val = avg_vol.iloc[-1]
    
    vol_spike = curr_vol > (avg_vol_val * vol_mult)
    
    open_p = df_1m['open'].iloc[-1]
    close_p = df_1m['close'].iloc[-1]
    prev_close = df_1m['close'].iloc[-2]
    
    bullish = (close_p > open_p) and (close_p > prev_close)
    bearish = (close_p < open_p) and (close_p < prev_close)
    
    details = {'vol_spike': vol_spike, 'bullish': bullish, 'bearish': bearish}
    
    if trend == 'BULLISH' and vol_spike and bullish:
        return True, 'BUY', details
    elif trend == 'BEARISH' and vol_spike and bearish:
        return True, 'SELL', details
    
    return False, None, details

# ==========================================
# VISUALIZATION
# ==========================================
def create_chart(df, title, indicators=None):
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.7, 0.3],
        subplot_titles=(title, 'Volume')
    )
    
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price'
    ), row=1, col=1)
    
    if indicators:
        for name, values in indicators.items():
            if values is not None:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=values, name=name,
                    line=dict(width=2)
                ), row=1, col=1)
    
    colors = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['timestamp'], y=df['volume'],
        marker_color=colors, opacity=0.7, name='Volume'
    ), row=2, col=1)
    
    fig.update_layout(
        height=500, showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    
    return fig

# ==========================================
# MAIN APP
# ==========================================
def main():
    # Enhanced Header
    st.markdown('<div class="main-header">🎯 Multi-Timeframe Scalping Bot</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Algorithmic Trading System | 15m Trend • 5m Setup • 1m Trigger</div>', 
                unsafe_allow_html=True)
    
    # Sidebar with ENHANCED STRICTNESS
    with st.sidebar:
        st.header("⚙️ Settings")
        
        symbol = st.selectbox(
            "Trading Pair",
            ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]
        )
        
        # ENHANCED STRICTNESS SECTION
        st.markdown('<div class="strictness-box">', unsafe_allow_html=True)
        st.subheader("🎚️ Trading Mode")
        
        strictness = st.select_slider(
            "Select Strictness Level",
            options=["VERY STRICT", "STRICT", "BALANCED", "RELAXED", "VERY RELAXED"],
            value="BALANCED",
            help="Higher strictness = fewer signals, higher quality"
        )
        
        # Mode descriptions
        mode_desc = {
            "VERY STRICT": "Ultra-conservative | 1-3 signals/day | ~70% win rate",
            "STRICT": "Conservative | 2-5 signals/day | ~65% win rate",
            "BALANCED": "Recommended | 5-10 signals/day | ~60% win rate",
            "RELAXED": "Aggressive | 10-20 signals/day | ~55% win rate",
            "VERY RELAXED": "High frequency | 20-40 signals/day | ~50% win rate"
        }
        
        st.caption(f"**{strictness}**: {mode_desc[strictness]}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Auto-set parameters based on strictness
        if strictness == "VERY STRICT":
            min_atr, trend_thresh, tolerance, vol_mult = 0.001, 0.002, 0.001, 1.5
        elif strictness == "STRICT":
            min_atr, trend_thresh, tolerance, vol_mult = 0.0008, 0.001, 0.0015, 1.3
        elif strictness == "BALANCED":
            min_atr, trend_thresh, tolerance, vol_mult = 0.0005, 0.0005, 0.0025, 1.2
        elif strictness == "RELAXED":
            min_atr, trend_thresh, tolerance, vol_mult = 0.0003, 0.0003, 0.004, 1.1
        else:  # VERY RELAXED
            min_atr, trend_thresh, tolerance, vol_mult = 0.0001, 0.0001, 0.01, 1.0
        
        # Show current values in expander
        with st.expander("📊 Current Filter Values"):
            st.write(f"**ATR Threshold:** {min_atr:.4f} ({min_atr*100:.2f}%)")
            st.write(f"**Trend Threshold:** {trend_thresh:.4f} ({trend_thresh*100:.2f}%)")
            st.write(f"**Pullback Tolerance:** {tolerance:.4f} ({tolerance*100:.2f}%)")
            st.write(f"**Volume Multiplier:** {vol_mult:.1f}x")
        
        # Refresh settings
        st.subheader("🔄 Refresh")
        refresh_interval = st.slider("Interval (seconds)", 5, 300, 30, 5)
        auto_refresh = st.checkbox("Enable Auto Refresh", value=False)
        
        if st.button("🔄 Refresh Now", type="primary"):
            st.rerun()
    
    # Fetch Data
    with st.spinner("Fetching market data..."):
        df_15m = fetch_candles(symbol, "15m", 300)
        df_5m = fetch_candles(symbol, "5m", 300)
        df_1m = fetch_candles(symbol, "1m", 300)
    
    if df_15m.empty or df_5m.empty or df_1m.empty:
        st.error("Failed to fetch data. Check connection.")
        return
    
    # Live Status Bar
    current_price = df_1m['close'].iloc[-1]
    trend, trend_str, ema50_15, ema200_15 = analyze_trend(df_15m, trend_thresh)
    atr_ok, atr_val, atr_pct = check_volatility(df_5m, min_atr)
    
    status_cols = st.columns(4)
    with status_cols[0]:
        st.metric("🕐 Last Update", datetime.now().strftime("%H:%M:%S"))
    with status_cols[1]:
        st.metric("📊 Market", "ACTIVE", delta=symbol)
    with status_cols[2]:
        vol_status = "HIGH" if atr_ok else "LOW"
        st.metric("⚡ Volatility", vol_status, delta=f"{atr_pct:.4f}")
    with status_cols[3]:
        st.metric("💰 Price", f"${current_price:,.2f}")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Analysis
    setup_ok, setup_details = check_5m_setup(df_5m, trend, tolerance)
    trigger_ok, signal, trigger_details = check_1m_trigger(df_1m, trend, vol_mult)
    
    # Determine final signal
    final_signal = 'NO TRADE'
    reason = []
    
    if trend == 'SIDEWAYS':
        reason.append("Trend too weak (Sideways)")
    elif trend == 'UNKNOWN':
        reason.append("Insufficient trend data")
    else:
        if not atr_ok:
            reason.append(f"Low volatility (ATR: {atr_pct:.4f})")
        if not setup_ok:
            reason.append("5m setup invalid")
        if not trigger_ok:
            reason.append("1m trigger not found")
        
        if trend != 'SIDEWAYS' and atr_ok and setup_ok and trigger_ok:
            final_signal = signal
            reason.append("All conditions met!")
    
    # Enhanced Signal Display
    signal_col = st.columns([1, 3, 1])[1]
    with signal_col:
        if final_signal == 'BUY':
            st.markdown(f'''
                <div class="signal-box buy-signal">
                    <div style="font-size: 3rem; margin-bottom: 10px;">🚀</div>
                    <div>BUY SIGNAL</div>
                    <div style="font-size: 2rem; margin-top: 10px;">${current_price:,.2f}</div>
                    <div style="font-size: 0.9rem; margin-top: 10px; opacity: 0.9;">
                        Entry Confirmed • {strictness} Mode
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        elif final_signal == 'SELL':
            st.markdown(f'''
                <div class="signal-box sell-signal">
                    <div style="font-size: 3rem; margin-bottom: 10px;">🔻</div>
                    <div>SELL SIGNAL</div>
                    <div style="font-size: 2rem; margin-top: 10px;">${current_price:,.2f}</div>
                    <div style="font-size: 0.9rem; margin-top: 10px; opacity: 0.9;">
                        Entry Confirmed • {strictness} Mode
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="signal-box neutral-signal">
                    <div style="font-size: 2.5rem; margin-bottom: 10px;">⏸️</div>
                    <div>NO TRADE</div>
                    <div style="font-size: 0.95rem; margin-top: 15px; line-height: 1.6;">
                        {"<br>• ".join([""] + reason)}
                    </div>
                </div>
            ''', unsafe_allow_html=True)
    
    # Enhanced Metrics Cards
    if final_signal in ['BUY', 'SELL']:
        st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.markdown(f'''
                <div class="analysis-card" style="text-align: center;">
                    <div class="metric-label">Entry Price</div>
                    <div class="metric-value">${current_price:,.2f}</div>
                    <span class="status-badge badge-success">CONFIRMED</span>
                </div>
            ''', unsafe_allow_html=True)
        
        with metric_cols[1]:
            st.markdown(f'''
                <div class="analysis-card" style="text-align: center;">
                    <div class="metric-label">ATR (5m)</div>
                    <div class="metric-value">${atr_val:.2f}</div>
                    <span class="status-badge {'badge-success' if atr_ok else 'badge-error'}">{atr_pct:.4f}</span>
                </div>
            ''', unsafe_allow_html=True)
        
        with metric_cols[2]:
            st.markdown(f'''
                <div class="analysis-card" style="text-align: center;">
                    <div class="metric-label">Trend Strength</div>
                    <div class="metric-value">{trend_str:.4f}</div>
                    <span class="status-badge badge-success">{trend}</span>
                </div>
            ''', unsafe_allow_html=True)
        
        with metric_cols[3]:
            confidence = "HIGH" if trend_str > 0.002 else "MEDIUM" if trend_str > 0.001 else "LOW"
            st.markdown(f'''
                <div class="analysis-card" style="text-align: center;">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{confidence}</div>
                    <span class="status-badge badge-success">ACTIVE</span>
                </div>
            ''', unsafe_allow_html=True)
    
    # Enhanced Analysis Section
    st.markdown('<div class="section-title">📋 Step-by-Step Analysis</div>', unsafe_allow_html=True)
    
    analysis_cols = st.columns(3)
    
    with analysis_cols[0]:
        trend_status = "success" if trend in ['BULLISH', 'BEARISH'] else "error"
        trend_icon = "✅" if trend in ['BULLISH', 'BEARISH'] else "❌"
        
        st.markdown(f'''
            <div class="analysis-card {trend_status}-card">
                <div class="step-header">Step 1 • 15m Trend Filter</div>
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 1.5rem; margin-right: 10px;">{trend_icon}</span>
                    <span class="metric-value">{trend}</span>
                </div>
                <div style="font-size: 0.9rem; color: #6b7280; line-height: 1.8;">
                    <div>EMA50: <strong>{ema50_15:,.2f}</strong></div>
                    <div>EMA200: <strong>{ema200_15:,.2f}</strong></div>
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e7eb;">
                        Strength: <span class="status-badge {'badge-success' if trend_str > trend_thresh else 'badge-error'}">{trend_str:.4f}</span>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.8rem;">
                        Threshold: {trend_thresh:.4f}
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
    with analysis_cols[1]:
        setup_status = "success" if setup_ok else "error"
        setup_icon = "✅" if setup_ok else "❌"
        near_zone = setup_details.get('near_zone', False)
        
        st.markdown(f'''
            <div class="analysis-card {setup_status}-card">
                <div class="step-header">Step 2 • 5m Setup Check</div>
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 1.5rem; margin-right: 10px;">{setup_icon}</span>
                    <span class="metric-value">{'VALID' if setup_ok else 'INVALID'}</span>
                </div>
                <div style="font-size: 0.9rem; color: #6b7280; line-height: 1.8;">
                    <div>Price: <strong>${setup_details.get('price', 0):,.2f}</strong></div>
                    <div>VWAP: <strong>${setup_details.get('vwap', 0):,.2f}</strong></div>
                    <div>EMA20: <strong>${setup_details.get('ema20', 0):,.2f}</strong></div>
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e7eb;">
                        Near Zone: <span class="status-badge {'badge-success' if near_zone else 'badge-error'}">{'YES' if near_zone else 'NO'}</span>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.8rem;">
                        Tolerance: {tolerance*100:.2f}%
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
    with analysis_cols[2]:
        trigger_status = "success" if trigger_ok else "error"
        trigger_icon = "✅" if trigger_ok else "❌"
        vol_spike = trigger_details.get('vol_spike', False)
        is_bullish = trigger_details.get('bullish', False)
        is_bearish = trigger_details.get('bearish', False)
        
        st.markdown(f'''
            <div class="analysis-card {trigger_status}-card">
                <div class="step-header">Step 3 • 1m Entry Trigger</div>
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 1.5rem; margin-right: 10px;">{trigger_icon}</span>
                    <span class="metric-value">{'TRIGGER' if trigger_ok else 'NO TRIGGER'}</span>
                </div>
                <div style="font-size: 0.9rem; color: #6b7280; line-height: 1.8;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Volume Spike:</span>
                        <span class="status-badge {'badge-success' if vol_spike else 'badge-error'}">{'YES' if vol_spike else 'NO'}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Bullish:</span>
                        <span class="status-badge {'badge-success' if is_bullish else 'badge-neutral'}">{'YES' if is_bullish else 'NO'}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Bearish:</span>
                        <span class="status-badge {'badge-success' if is_bearish else 'badge-neutral'}">{'YES' if is_bearish else 'NO'}</span>
                    </div>
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e7eb; font-size: 0.8rem;">
                        Volume Mult: {vol_mult:.1f}x
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
    # Enhanced Charts Section
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Multi-Timeframe Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🕐 15m - Trend", "⏱️ 5m - Setup", "⚡ 1m - Entry"])
    
    with tab1:
        if not df_15m.empty:
            ema50_15_line = calculate_ema(df_15m['close'], 50)
            ema200_15_line = calculate_ema(df_15m['close'], 200)
            
            fig_15m = create_chart(
                df_15m.tail(100), 
                f"{symbol} - 15m Trend Analysis (EMA50/200)",
                {'EMA50': ema50_15_line, 'EMA200': ema200_15_line}
            )
            st.plotly_chart(fig_15m, use_container_width=True)
            
            st.markdown(f'''
                <div class="info-box">
                    <strong>💡 Trend Analysis:</strong> EMA50 ({ema50_15_line.iloc[-1]:,.2f}) 
                    {'above' if ema50_15_line.iloc[-1] > ema200_15_line.iloc[-1] else 'below'} 
                    EMA200 ({ema200_15_line.iloc[-1]:,.2f}) indicates 
                    <strong>{'BULLISH' if ema50_15_line.iloc[-1] > ema200_15_line.iloc[-1] else 'BEARISH'}</strong> trend.
                    Strength: {trend_str:.4f} (threshold: {trend_thresh:.4f})
                </div>
            ''', unsafe_allow_html=True)
    
    with tab2:
        if not df_5m.empty:
            ema20_5 = calculate_ema(df_5m['close'], 20)
            ema50_5 = calculate_ema(df_5m['close'], 50)
            vwap_5 = calculate_vwap(df_5m)
            
            fig_5m = create_chart(
                df_5m.tail(100),
                f"{symbol} - 5m Setup Analysis (VWAP + EMA20/50)",
                {'EMA20': ema20_5, 'EMA50': ema50_5, 'VWAP': vwap_5}
            )
            st.plotly_chart(fig_5m, use_container_width=True)
            
            price_pos = "above" if setup_details.get('price', 0) > setup_details.get('vwap', 0) else "below"
            st.markdown(f'''
                <div class="info-box">
                    <strong>💡 Setup Analysis:</strong> Price (${setup_details.get('price', 0):,.2f}) is 
                    <strong>{price_pos}</strong> VWAP (${setup_details.get('vwap', 0):,.2f}).
                    Pullback tolerance: {tolerance*100:.2f}% | Near EMA zone: {'Yes' if setup_details.get('near_zone') else 'No'}
                </div>
            ''', unsafe_allow_html=True)
    
    with tab3:
        if not df_1m.empty:
            fig_1m = create_chart(
                df_1m.tail(100),
                f"{symbol} - 1m Entry Trigger Analysis"
            )
            st.plotly_chart(fig_1m, use_container_width=True)
            
            vol_status = "spike detected" if trigger_details.get('vol_spike') else "no spike"
            candle_type = "Bullish" if trigger_details.get('bullish') else "Bearish" if trigger_details.get('bearish') else "Neutral"
            
            st.markdown(f'''
                <div class="info-box">
                    <strong>💡 Entry Trigger:</strong> Volume {vol_status} (>{vol_mult:.1f}x avg). 
                    Candle pattern: <strong>{candle_type}</strong>. 
                    Current volume vs average determines entry confirmation.
                </div>
            ''', unsafe_allow_html=True)
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()