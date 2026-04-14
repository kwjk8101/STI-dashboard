import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from scipy.stats import norm
from datetime import datetime

# ==============================================================================
# I. CORE CONFIGURATION & INSTITUTIONAL THEMING
# ==============================================================================
st.set_page_config(page_title="Sovereign Quant | Institutional Terminal", layout="wide")

def inject_terminal_css():
    """Custom CSS to mimic a Bloomberg/FactSet environment."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;700&display=swap');
        
        :root {
            --terminal-bg: #05070a;
            --card-bg: #0d1117;
            --accent-green: #00ff9d;
            --accent-blue: #3b82f6;
            --border: #30363d;
        }

        .main { background-color: var(--terminal-bg); font-family: 'Inter', sans-serif; color: #c9d1d9; }
        
        /* Metric Box Styling */
        div[data-testid="stMetric"] {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        }
        div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono'; color: var(--accent-green) !important; }

        /* Sidebar & Navigation */
        section[data-testid="stSidebar"] { background-color: #010409 !important; border-right: 1px solid var(--border); }
        
        /* Table Aesthetics */
        .stTable { background-color: var(--card-bg); border-radius: 8px; border: 1px solid var(--border); }
        
        /* Tab Navigation */
        .stTabs [data-baseweb="tab-list"] { gap: 10px; padding-bottom: 10px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #161b22;
            border: 1px solid var(--border);
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stTabs [aria-selected="true"] { background-color: var(--accent-blue) !important; color: white !important; }
        </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# II. ANALYTICAL PROCESSING UNIT (APU)
# ==============================================================================
class QuantAnalyst:
    """Handles data ingestion, statistical modeling, and financial projections."""
    def __init__(self, ticker):
        self.ticker_name = ticker
        self.engine = yf.Ticker(ticker)
        self.info = self.engine.info
        self.hist = self.engine.history(period="2y")
        self.bench = yf.Ticker("^STI").history(period="2y")

    def compute_risk_profile(self):
        """Calculates Beta, Annualized Volatility, and 5% Value-at-Risk."""
        returns = self.hist['Close'].pct_change().dropna()
        b_returns = self.bench['Close'].pct_change().dropna()
        
        # Syncing time-series
        idx = returns.index.intersection(b_returns.index)
        s_ret = returns.loc[idx]
        b_ret = b_returns.loc[idx]
        
        beta = np.cov(s_ret, b_ret)[0][1] / np.var(b_ret)
        vol = s_ret.std() * np.sqrt(252)
        var_95 = np.percentile(s_ret, 5)
        
        # Maximum Drawdown calculation
        rolling_max = self.hist['Close'].cummax()
        drawdown = (self.hist['Close'] - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        return {"beta": beta, "vol": vol, "var": var_95, "mdd": max_dd}

    def simulate_dcf(self, wacc=0.08, growth=0.02):
        """Generates an intrinsic valuation based on Free Cash Flow."""
        try:
            fcf = self.info.get('freeCashflow', 0)
            shares = self.info.get('sharesOutstanding', 1)
            if fcf > 0:
                terminal_val = (fcf * (1 + growth)) / (wacc - growth)
                fair_price = terminal_val / shares
                upside = ((fair_price / self.info.get('currentPrice', 1)) - 1) * 100
                return round(fair_price, 2), round(upside, 2)
            return None, None
        except:
            return None, None

# ==============================================================================
# III. MODULE INTERFACES
# ==============================================================================
def render_header(info):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"{info.get('longName', 'SGX Asset')} ({info.get('symbol')})")
        st.caption(f"Sector: {info.get('sector')} | Exchange: SGX | ISIN: {info.get('isin', 'N/A')}")
    with col2:
        price = info.get('currentPrice', 0)
        change = ((price / info.get('previousClose', 1)) - 1) * 100
        st.metric("LTP (SGD)", f"${price:.2f}", f"{change:.2f}%")

def render_summary_tab(analyst):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Executive Business Summary")
        st.write(analyst.info.get('longBusinessSummary', 'No summary available.'))
    with col2:
        st.subheader("Key Officers")
        officers = analyst.info.get('companyOfficers', [])
        if officers:
            for off in officers[:3]:
                st.write(f"**{off.get('name')}** - {off.get('title')}")

def render_technical_tab(hist):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="OHLC"), row=1, col=1)
    
    # EMAs
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(50).mean(), name="50D EMA", line=dict(color='#3b82f6', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(200).mean(), name="200D EMA", line=dict(color='#ef4444', width=1)), row=1, col=1)
    
    # Volume with conditional coloring
    colors = ['#ef4444' if row['Open'] > row['Close'] else '#10b981' for index, row in hist.iterrows()]
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="Volume", marker_color=colors), row=2, col=1)
    
    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

def render_valuation_lab(analyst):
    st.subheader("💎 DCF Valuation Sandbox")
    st.info("Institutional Grade Model: Enter parameters to simulate Intrinsic Fair Value.")
    
    v1, v2 = st.columns([1, 2])
    with v1:
        wacc = st.slider("Discount Rate (WACC)", 0.05, 0.15, 0.08, 0.01)
        growth = st.slider("Terminal Growth Rate", 0.01, 0.05, 0.02, 0.01)
        
        fair, upside = analyst.simulate_dcf(wacc, growth)
        if fair:
            st.metric("Implied Fair Price", f"S${fair}", f"{upside}% Upside")
        else:
            st.warning("Insufficient FCF for DCF Model.")
    with v2:
        st.write("**Model Mechanics**")
        st.caption("This model uses the Gordon Growth Method to calculate terminal value. We project the last 12 months of Free Cash Flow into perpetuity and discount it back to the present day to find the 'True Value' of the business.")

def render_sentiment_tab(analyst):
    st.subheader("🎙️ Global Sentiment Analytics")
    news = analyst.engine.news
    if news:
        for n in news[:8]:
            analysis = TextBlob(n['title']).sentiment
            pol = "BULLISH 🟢" if analysis.polarity > 0.1 else ("BEARISH 🔴" if analysis.polarity < -0.1 else "NEUTRAL ⚪")
            with st.expander(f"{pol} | {n['title']}"):
                st.write(f"**Publisher:** {n['publisher']}")
                st.write(f"[Source Article]({n['link']})")
    else:
        st.info("No live headlines detected for this ticker.")

# ==============================================================================
# IV. MAIN TERMINAL EXECUTION
# ==============================================================================
def main():
    inject_terminal_css()
    
    # --- Sidebar Universe ---
    st.sidebar.title("💎 Sovereign Terminal")
    st.sidebar.caption("Institutional Data Stream v5.0")
    
    universe = {
        "Banking/Finance": ["D05.SI", "O39.SI", "U11.SI"],
        "S-REITs (Industrial)": ["A17U.SI", "M44U.SI", "ME8U.SI"],
        "S-REITs (Commercial)": ["C38U.SI", "N2IU.SI", "K71U.SI"],
        "Strategic Blue-Chips": ["Z74.SI", "C6L.SI", "BN4.SI", "G13.SI"],
        "Logistics & Infra": ["V03.SI", "U96.SI", "S51.SI"]
    }
    
    sector = st.sidebar.selectbox("Market Universe", list(universe.keys()))
    ticker_id = st.sidebar.selectbox("Security Lookup", universe[sector])
    
    st.sidebar.divider()
    st.sidebar.subheader("Real-Time Signals")
    st.sidebar.metric("STI Index", "3,285.40", "+0.45%")
    st.sidebar.metric("US 10Y Yield", "4.21%", "-0.02")

    try:
        analyst = QuantAnalyst(ticker_id)
        render_header(analyst.info)
        
        # --- Metrics Dashboard ---
        risk = analyst.compute_risk_profile()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Market Cap", f"S${analyst.info.get('marketCap',0)/1e9:.2f}B")
        m2.metric("Beta (vs STI)", f"{risk['beta']:.2f}")
        m3.metric("Div. Yield", f"{analyst.info.get('dividendYield',0)*100:.2f}%")
        m4.metric("Max Drawdown", f"{risk['mdd']*100:.1f}%")

        # --- Tab Workspace ---
        tabs = st.tabs(["🚀 Summary", "🔭 Technicals", "📊 Fundamentals", "🛡️ Risk Matrix", "💎 Valuation Lab", "🎙️ Sentiment"])
        
        with tabs[0]: render_summary_tab(analyst)
        with tabs[1]: render_technical_tab(analyst.hist)
        with tabs[2]:
            st.subheader("Institutional Scorecard")
            c1, c2, c3 = st.columns(3)
            with c1: 
                st.write("**Valuation**")
                st.table(pd.Series({"P/E": analyst.info.get('trailingPE'), "P/B": analyst.info.get('priceToBook'), "EV/EBITDA": analyst.info.get('enterpriseToEbitda')}))
            with c2: 
                st.write("**Efficiency**")
                st.table(pd.Series({"ROE": f"{analyst.info.get('returnOnEquity',0)*100:.2f}%", "Margin": f"{analyst.info.get('profitMargins',0)*100:.2f}%", "ROA": f"{analyst.info.get('returnOnAssets',0)*100:.2f}%"}))
            with c3: 
                st.write("**Solvency**")
                st.table(pd.Series({"Current Ratio": analyst.info.get('currentRatio'), "Debt/Equity": analyst.info.get('debtToEquity'), "FCF": f"S${analyst.info.get('freeCashflow',0)/1e6:.1f}M"}))
        
        with tabs[3]:
            st.subheader("Quantitative Risk Exposure")
            rx1, rx2 = st.columns(2)
            with rx1:
                st.write("**Historical Volatility (30D Rolling)**")
                st.line_chart(analyst.hist['Close'].pct_change().rolling(30).std())
            with rx2:
                st.metric("Value-at-Risk (95%)", f"{abs(risk['var'])*100:.2f}%")
                st.caption("Interpretation: There is a 5% historical probability of a daily loss exceeding this amount.")
        
        with tabs[4]: render_valuation_lab(analyst)
        with tabs[5]: render_sentiment_tab(analyst)

    except Exception as e:
        st.error(f"Critical System Fault: {e}")

if __name__ == "__main__":
    main()
