# TradingBot

A comprehensive trading bot for forex markets that connects to the XTB API (or runs in simulation mode) to execute trades based on technical indicators.

## Features

- **Real-time Trading**: Connects to XTB's demo API for live trading
- **Simulation Mode**: Backtest strategies with historical data
- **Technical Indicators**: 
  - Supertrend
  - RSI
  - MACD
  - Bollinger Bands
  - Ichimoku Cloud
  - Parabolic SAR
  - Moving Averages
  - Fibonacci Levels
- **Risk Management**: Position sizing based on account balance and risk percentage
- **Multi-Timeframe Analysis**: Supports various candle periods (M1 to MN1)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rickil/TradingBot.git
cd TradingBot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your configuration in `config.py`:
```python
base_url = "wss://ws.xtb.com/demo"  # Demo server URL
userEmail = "your@email.com"        # Your XTB email
userId = "your_user_id"             # Your XTB user ID
password = "your_password"          # Your XTB password
```

## Usage

### Live Trading
```python
from Bot import Bot
from config import base_url, userId, password

bot = Bot(name="LiveBot", 
          url=base_url, 
          userId=userId, 
          password=password,
          period="H1", 
          qty_candles=500)
bot.run()
```

### Simulation Mode
```python
from Bot import Bot

bot = Bot(name="SimBot",
          url="", 
          userId="", 
          password="",
          simulation=True,
          simulation_data_file="samples/forex_H1.json",
          sample_size=1000,
          initial_balance=10000)
bot.run()
```

### Generating Sample Data
Run the samples generator to create historical data files:
```bash
python samples_generator.py
```

## File Structure

- `API.py`: XTB API connection and methods
- `Bot.py`: Main trading bot logic
- `Order.py`: Order management system
- `SignalDetector.py`: Technical analysis and signal generation
- `Simulation.py`: Backtesting simulation environment
- `config.py`: Configuration file for API credentials
- `samples_generator.py`: Script to generate historical data samples
- `simulation_main.py`: Example simulation script

## Indicators Implemented

The bot uses multiple technical indicators to generate trading signals:

1. **Engulfing Patterns** (Bullish/Bearish)
2. **Supertrend** (3 variations with different parameters)
3. **RSI** (Overbought/Oversold conditions)
4. **MACD** (Crossovers)
5. **Bollinger Bands** (Price breakout and squeeze)
6. **Ichimoku Cloud** (Tenkan/Kijun cross)
7. **Parabolic SAR** (Trend reversal)
8. **Moving Averages** (Golden/Death Cross)
9. **Fibonacci Levels** (Support/Resistance)

## Risk Management

The bot includes position sizing based on:
- Account balance
- Symbol margin requirements
- Configurable risk percentage (default 1%)
- Minimum/maximum lot sizes

## Performance Metrics

In simulation mode, the bot tracks:
- Total trades
- Winning/losing trades
- Success percentage
- Average profit/loss
- Final balance
- Profit/loss summary

## Requirements

- Python 3.7+
- websocket-client
- pandas
- numpy
- openpyxl
- tqdm
