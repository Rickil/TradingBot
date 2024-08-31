import pandas as pd
import numpy as np
from enum import Enum
from Order import ORDER_TYPE

class Indicator:
    def isEngulfing(self, data):
        prev_open = data[-3]["open"]
        prev_close = data[-3]["close"]
        curr_open = data[-2]["open"]
        curr_close = data[-2]["close"]

        if prev_close < prev_open and curr_close > curr_open and curr_open < prev_close and curr_close > prev_open:
            return True
        
        if prev_close > prev_open and curr_close < curr_open and curr_open > prev_close and curr_close < prev_open:
            return True
        
        return False

    def calculate_atr(self, df, period):
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = abs(df['low'] - df['close'].shift(1))
        
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=period).mean()
        
        return df['ATR']

    def calculate_supertrend(self, data, period=10, multiplier=2):
        df = pd.DataFrame(data)
        df['ATR'] = self.calculate_atr(df, period)
        
        df['Upper Band'] = ((df['high'] + df['low']) / 2) + (multiplier * df['ATR'])
        df['Lower Band'] = ((df['high'] + df['low']) / 2) - (multiplier * df['ATR'])
        
        df['Supertrend'] = 0.0
        df.loc[0, 'Supertrend'] = df.loc[0, 'Upper Band'] if df.loc[0, 'close'] > df.loc[0, 'Upper Band'] else df.loc[0, 'Lower Band']
        
        for i in range(1, len(df)):
            if df.loc[i-1, 'close'] > df.loc[i-1, 'Supertrend']:
                df.loc[i, 'Supertrend'] = df.loc[i, 'Lower Band'] if df.loc[i, 'close'] > df.loc[i, 'Lower Band'] else df.loc[i, 'Upper Band']
            else:
                df.loc[i, 'Supertrend'] = df.loc[i, 'Upper Band'] if df.loc[i, 'close'] < df.loc[i, 'Upper Band'] else df.loc[i, 'Lower Band']
        
        return df['Supertrend']

    def bullish_engulfing(self, data):
        if data[-2]["type"] == "Bullish" and data[-3]["type"] == "Bearish" and self.isEngulfing(data):
            return True
        else:
            return False
    
    def bearish_engulfing(self, data):
        if data[-2]["type"] == "Bearish" and data[-3]["type"] == "Bullish" and self.isEngulfing(data):
            return True
        else:
            return False
        
    def calculate_moving_average(self, df, period):
        return df['close'].rolling(window=period).mean()

    def calculate_rsi(self, df, period=14):
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, df, short_period=12, long_period=26, signal_period=9):
        short_ema = df['close'].ewm(span=short_period, adjust=False).mean()
        long_ema = df['close'].ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    def calculate_bollinger_bands(self, df, period=20, std_multiplier=2):
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper_band = sma + (std_multiplier * std)
        lower_band = sma - (std_multiplier * std)
        return upper_band, lower_band

    def calculate_fibonacci_levels(self, df):
        max_price = df['close'].max()
        min_price = df['close'].min()
        diff = max_price - min_price
        levels = {
            '23.6%': max_price - 0.236 * diff,
            '38.2%': max_price - 0.382 * diff,
            '50%': max_price - 0.5 * diff,
            '61.8%': max_price - 0.618 * diff,
            '100%': min_price
        }
        return levels
    
    def calculate_ichimoku_cloud(self, df):
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2

        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        kijun_sen = (high_26 + low_26) / 2

        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)

        chikou_span = df['close'].shift(-26)

        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

    def calculate_parabolic_sar(self, df, step=0.02, max_step=0.2):
        sar = pd.Series(df['low'][0:2].min(), index=df.index)
        long = True
        af = step
        ep = df['high'][0]

        for i in range(2, len(df)):
            if long:
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                sar[i] = min(sar[i], df['low'][i - 1], df['low'][i - 2])
                if df['low'][i] < sar[i]:
                    long = False
                    sar[i] = ep
                    af = step
                    ep = df['low'][i]
            else:
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                sar[i] = max(sar[i], df['high'][i - 1], df['high'][i - 2])
                if df['high'][i] > sar[i]:
                    long = True
                    sar[i] = ep
                    af = step
                    ep = df['high'][i]
            if long:
                if df['high'][i] > ep:
                    ep = df['high'][i]
                    af = min(af + step, max_step)
            else:
                if df['low'][i] < ep:
                    ep = df['low'][i]
                    af = min(af + step, max_step)

        return sar
    
    def detect_golden_cross(self, short_ma, long_ma):
        return short_ma.iloc[-2] < long_ma.iloc[-2] and short_ma.iloc[-1] > long_ma.iloc[-1]

    def detect_death_cross(self, short_ma, long_ma):
        return short_ma.iloc[-2] > long_ma.iloc[-2] and short_ma.iloc[-1] < long_ma.iloc[-1]

    def detect_rsi_divergence(self, df, rsi):
        bullish_divergence = df['close'].iloc[-1] < df['close'].iloc[-2] and rsi.iloc[-1] > rsi.iloc[-2]
        bearish_divergence = df['close'].iloc[-1] > df['close'].iloc[-2] and rsi.iloc[-1] < rsi.iloc[-2]
        return bullish_divergence, bearish_divergence

    def detect_price_breakout(self, df):
        recent_high = df['high'].rolling(window=20).max().iloc[-1]
        recent_low = df['low'].rolling(window=20).min().iloc[-1]
        breakout_up = df['close'].iloc[-1] > recent_high
        breakout_down = df['close'].iloc[-1] < recent_low
        return breakout_up, breakout_down

    def detect_bollinger_squeeze(self, upper_band, lower_band):
        bandwidth = (upper_band - lower_band) / upper_band
        return bandwidth.iloc[-1] < 0.05

class SIGNAL_TYPE(Enum):
    OPEN = 0
    CLOSE = 2

class Signal:
    def __init__(self, ORDER_TYPE, signal_type, volume=0):
        self.cmd = ORDER_TYPE
        self.type = signal_type
        self.volume = volume

class SignalDetector:
    def __init__(self):
        self.indicator = Indicator()

    def calculate_trade_volume(self, balance, symbolInfos, risk_percentage=1):
        """
        Calculate the trade volume based on the balance, leverage, and risk management.
        The balance is in EUR, and we use the lotMinMargin field from symbolInfos.
        The function also ensures that the balance is sufficient for the calculated volume.
        """
        # Calculate the amount of money we are willing to risk
        risk_amount = balance * (risk_percentage / 100)

        # Use the lotMinMargin directly from the symbolInfos
        lot_min_margin = symbolInfos['lotMinMargin']

        # Calculate the initial volume based on the risk amount and minimum lot margin
        volume = (risk_amount / lot_min_margin) * symbolInfos['lotMin']

        # Ensure volume respects broker's min, max, and step constraints
        volume = max(min(volume, symbolInfos['lotMax']), symbolInfos['lotMin'])
        volume = round(volume / symbolInfos['lotStep']) * symbolInfos['lotStep']

        # Check if the calculated volume requires more margin than available in the balance
        required_margin = volume * lot_min_margin / symbolInfos['lotMin']
        if required_margin > balance:
            # If the required margin exceeds the balance, adjust the volume downwards
            volume = (balance / lot_min_margin) * symbolInfos['lotMin']
            volume = max(min(volume, symbolInfos['lotMax']), symbolInfos['lotMin'])
            volume = round(volume / symbolInfos['lotStep']) * symbolInfos['lotStep']
        
        #check that required margin is less than balance
        required_margin = volume * lot_min_margin / symbolInfos['lotMin']
        if required_margin > balance:
            return 0

        return volume



    def check_enterTrade_signal(self, data, balance, symbolInfos):
        df = pd.DataFrame(data)
        
        # Calculate indicators
        supertrend1 = self.indicator.calculate_supertrend(data, period=10, multiplier=2)
        supertrend2 = self.indicator.calculate_supertrend(data, period=12, multiplier=3)
        supertrend3 = self.indicator.calculate_supertrend(data, period=11, multiplier=2)
        rsi = self.indicator.calculate_rsi(df, period=14)
        macd, signal = self.indicator.calculate_macd(df, short_period=12, long_period=26, signal_period=9)
        upper_band, lower_band = self.indicator.calculate_bollinger_bands(df, period=20, std_multiplier=2)
        tenkan_sen, kijun_sen, _, _, _ = self.indicator.calculate_ichimoku_cloud(df)
        sar = self.indicator.calculate_parabolic_sar(df, step=0.02, max_step=0.2)
        short_ma = self.indicator.calculate_moving_average(df, period=50)
        long_ma = self.indicator.calculate_moving_average(df, period=200)

        # Determine the volume for the trade
        volume = self.calculate_trade_volume(balance, symbolInfos)
        if volume == 0:
            return None

        # Buy Conditions
        if self.indicator.bullish_engulfing(data):
            if (
                supertrend1.iloc[-1] > supertrend1.iloc[-2] and
                supertrend2.iloc[-1] > supertrend2.iloc[-2] and
                supertrend3.iloc[-1] > supertrend3.iloc[-2]
            ):
                return Signal(ORDER_TYPE.BUY,  SIGNAL_TYPE.OPEN, volume)
                

        if rsi.iloc[-1] < 30:  # RSI oversold
            return Signal(ORDER_TYPE.BUY,  SIGNAL_TYPE.OPEN, volume)
            
        
        if macd.iloc[-1] > signal.iloc[-1]:  # MACD bullish crossover
            return Signal(ORDER_TYPE.BUY,  SIGNAL_TYPE.OPEN, volume)
            

        if df['close'].iloc[-1] < lower_band.iloc[-1]:  # Price below Bollinger Band
            return Signal(ORDER_TYPE.BUY,  SIGNAL_TYPE.OPEN, volume)
            

        if df['close'].iloc[-1] > tenkan_sen.iloc[-1] and df['close'].iloc[-1] > kijun_sen.iloc[-1]:  # Ichimoku Cloud bullish signal
            return Signal(ORDER_TYPE.BUY,  SIGNAL_TYPE.OPEN, volume)
            

        if df['close'].iloc[-1] > sar.iloc[-1]:  # Parabolic SAR buy signal
            return Signal(ORDER_TYPE.BUY,  SIGNAL_TYPE.OPEN, volume)
            

        # Sell Conditions
        if self.indicator.bearish_engulfing(data):
            if (
                supertrend1.iloc[-1] < supertrend1.iloc[-2] and
                supertrend2.iloc[-1] < supertrend2.iloc[-2] and
                supertrend3.iloc[-1] < supertrend3.iloc[-2]
            ):
                return Signal(ORDER_TYPE.SELL,  SIGNAL_TYPE.OPEN, volume)
                

        if rsi.iloc[-1] > 70:  # RSI overbought
            return Signal(ORDER_TYPE.SELL,  SIGNAL_TYPE.OPEN, volume)
            

        if macd.iloc[-1] < signal.iloc[-1]:  # MACD bearish crossover
            return Signal(ORDER_TYPE.SELL,  SIGNAL_TYPE.OPEN, volume)
            

        if df['close'].iloc[-1] > upper_band.iloc[-1]:  # Price above Bollinger Band
            return Signal(ORDER_TYPE.SELL,  SIGNAL_TYPE.OPEN, volume)
            

        if df['close'].iloc[-1] < tenkan_sen.iloc[-1] and df['close'].iloc[-1] < kijun_sen.iloc[-1]:  # Ichimoku Cloud bearish signal
            return Signal(ORDER_TYPE.SELL,  SIGNAL_TYPE.OPEN, volume)
            

        if df['close'].iloc[-1] < sar.iloc[-1]:  # Parabolic SAR sell signal
            return Signal(ORDER_TYPE.SELL,  SIGNAL_TYPE.OPEN, volume)
            
        
        # Golden Cross / Death Cross
        if self.indicator.detect_golden_cross(short_ma, long_ma):
            return Signal(ORDER_TYPE.BUY,  SIGNAL_TYPE.OPEN, volume)
            
        if self.indicator.detect_death_cross(short_ma, long_ma):
            return Signal(ORDER_TYPE.SELL,  SIGNAL_TYPE.OPEN, volume)
            

        # RSI Divergence
        bullish_divergence, bearish_divergence = self.indicator.detect_rsi_divergence(df, rsi)
        if bullish_divergence:
            return Signal(ORDER_TYPE.BUY,  SIGNAL_TYPE.OPEN, volume)
            
        if bearish_divergence:
            return Signal(ORDER_TYPE.SELL,  SIGNAL_TYPE.OPEN, volume)
            

        # Breakout Trading
        breakout_up, breakout_down = self.indicator.detect_price_breakout(df)
        if breakout_up:
            return Signal(ORDER_TYPE.BUY,  SIGNAL_TYPE.OPEN, volume)
            
        if breakout_down:
            return Signal(ORDER_TYPE.SELL,  SIGNAL_TYPE.OPEN, volume)
            

        # Bollinger Bands Squeeze
        if self.indicator.detect_bollinger_squeeze(upper_band, lower_band):
            if breakout_up:
                return Signal(ORDER_TYPE.BUY,  SIGNAL_TYPE.OPEN, volume)
                
            if breakout_down:
                return Signal(ORDER_TYPE.SELL,  SIGNAL_TYPE.OPEN, volume)
                

        return None

    def check_closeTrade_signal(self, data, order):
        df = pd.DataFrame(data)

        # Calculate indicators
        supertrend1 = self.indicator.calculate_supertrend(data, period=10, multiplier=2)
        supertrend2 = self.indicator.calculate_supertrend(data, period=12, multiplier=3)
        supertrend3 = self.indicator.calculate_supertrend(data, period=11, multiplier=2)
        rsi = self.indicator.calculate_rsi(df, period=14)
        macd, signal = self.indicator.calculate_macd(df, short_period=12, long_period=26, signal_period=9)
        upper_band, lower_band = self.indicator.calculate_bollinger_bands(df, period=20, std_multiplier=2)
        tenkan_sen, kijun_sen, _, _, _ = self.indicator.calculate_ichimoku_cloud(df)
        sar = self.indicator.calculate_parabolic_sar(df, step=0.02, max_step=0.2)
        short_ma = self.indicator.calculate_moving_average(df, period=50)
        long_ma = self.indicator.calculate_moving_average(df, period=200)
        atr = self.indicator.calculate_atr(df, period=14)

        current_price = df['open'].iloc[-1]

        # Exit conditions for buy signals
        if order.type == ORDER_TYPE.BUY:
            # Supertrend exit
            if (
                supertrend1.iloc[-1] < supertrend1.iloc[-2] or
                supertrend2.iloc[-1] < supertrend2.iloc[-2] or
                supertrend3.iloc[-1] < supertrend3.iloc[-2]
            ):
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # RSI Overbought
            if rsi.iloc[-1] > 70:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # MACD Bearish Crossover
            if macd.iloc[-1] < signal.iloc[-1]:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # Price above Bollinger Upper Band
            if df['close'].iloc[-1] > upper_band.iloc[-1]:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # Ichimoku Cloud Bearish Signal
            if df['close'].iloc[-1] < tenkan_sen.iloc[-1] and df['close'].iloc[-1] < kijun_sen.iloc[-1]:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # Parabolic SAR sell signal
            if df['close'].iloc[-1] < sar.iloc[-1]:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # Moving Average Exit (Price crosses below short-term MA)
            if df['close'].iloc[-1] < short_ma.iloc[-1]:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # ATR-Based Exit (Trailing Stop)
            if order.entry_price and current_price < order.entry_price - 2 * atr.iloc[-1]:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # Fibonacci Extensions Exit (when price reaches Fibonacci extension levels)
            fib_levels = self.indicator.calculate_fibonacci_levels(df)
            if current_price >= fib_levels['100%']:  # Target reached
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

        # Exit conditions for sell signals
        if order.type == ORDER_TYPE.SELL:
            # Supertrend exit
            if (
                supertrend1.iloc[-1] > supertrend1.iloc[-2] or
                supertrend2.iloc[-1] > supertrend2.iloc[-2] or
                supertrend3.iloc[-1] > supertrend3.iloc[-2]
            ):
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # RSI Oversold
            if rsi.iloc[-1] < 30:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # MACD Bullish Crossover
            if macd.iloc[-1] > signal.iloc[-1]:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # Price below Bollinger Lower Band
            if df['close'].iloc[-1] < lower_band.iloc[-1]:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # Ichimoku Cloud Bullish Signal
            if df['close'].iloc[-1] > tenkan_sen.iloc[-1] and df['close'].iloc[-1] > kijun_sen.iloc[-1]:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # Parabolic SAR buy signal
            if df['close'].iloc[-1] > sar.iloc[-1]:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # Moving Average Exit (Price crosses above short-term MA)
            if df['close'].iloc[-1] > short_ma.iloc[-1]:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # ATR-Based Exit (Trailing Stop)
            if order.entry_price and current_price > order.entry_price + 2 * atr.iloc[-1]:
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

            # Fibonacci Extensions Exit (when price retraces after hitting an extension level)
            fib_levels = self.indicator.calculate_fibonacci_levels(df)
            if current_price <= fib_levels['100%']:  # Target reached
                return Signal(order.type,  SIGNAL_TYPE.CLOSE, order.volume)
                

        return None