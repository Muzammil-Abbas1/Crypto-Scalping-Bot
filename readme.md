# 🎯 Multi-Timeframe Crypto Scalping Bot

A real-time algorithmic trading system that analyzes cryptocurrency markets across multiple timeframes to generate high-probability trade signals.

![Bot Screenshot](images/project-crypto-bot.png)

## ⚡ Features

- **3-Layer Signal Filtering**: 15m trend → 5m setup → 1m entry trigger
- **Technical Indicators**: EMA, VWAP, ATR, Volume analysis
- **5 Strictness Modes**: From ultra-conservative to high-frequency trading
- **Real-time Charts**: Interactive Plotly visualizations
- **Web Dashboard**: Clean Streamlit interface

## 🚀 Quick Start

```bash
# Install dependencies
pip install streamlit ccxt pandas numpy plotly

# Run the bot
streamlit run app.py

#live test
