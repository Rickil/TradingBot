import pandas as pd
import numpy as np

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

class Signal:
    def __init__(self, type, volume=0):
        self.type = type
        self.volume = volume

class SignalDetector:
    def __init__(self):
        self.signal = None
        self.currency_infos = None
        self.balance = None
        self.indicator = Indicator()

    def calculate_trade_volume(self, risk_percentage=1):
        """
        Calculate the trade volume based on the balance, leverage, and risk management.
        """
        risk_amount = self.balance * (risk_percentage / 100)
        pip_value = (self.currency_infos['tickValue'] * self.currency_infos['contractSize']) / self.currency_infos['leverage']
        
        # Assuming a pip move equals tickSize, calculate volume to risk a certain amount
        volume = risk_amount / (pip_value * self.currency_infos['tickSize'])
        
        # Ensure volume respects broker's min, max, and step constraints
        volume = max(min(volume, self.currency_infos['lotMax']), self.currency_infos['lotMin'])
        volume = round(volume / self.currency_infos['lotStep']) * self.currency_infos['lotStep']
        
        return volume

    def check_enterTrade_signal(self, data):
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
        volume = self.calculate_trade_volume()

        # Buy Conditions
        if self.indicator.bullish_engulfing(data):
            if (
                supertrend1.iloc[-1] > supertrend1.iloc[-2] and
                supertrend2.iloc[-1] > supertrend2.iloc[-2] and
                supertrend3.iloc[-1] > supertrend3.iloc[-2]
            ):
                self.signal = Signal("buy", volume)
                return self.signal

        if rsi.iloc[-1] < 30:  # RSI oversold
            self.signal = Signal("buy", volume)
            return self.signal
        
        if macd.iloc[-1] > signal.iloc[-1]:  # MACD bullish crossover
            self.signal = Signal("buy", volume)
            return self.signal

        if df['close'].iloc[-1] < lower_band.iloc[-1]:  # Price below Bollinger Band
            self.signal = Signal("buy", volume)
            return self.signal

        if df['close'].iloc[-1] > tenkan_sen.iloc[-1] and df['close'].iloc[-1] > kijun_sen.iloc[-1]:  # Ichimoku Cloud bullish signal
            self.signal = Signal("buy", volume)
            return self.signal

        if df['close'].iloc[-1] > sar.iloc[-1]:  # Parabolic SAR buy signal
            self.signal = Signal("buy", volume)
            return self.signal

        # Sell Conditions
        if self.indicator.bearish_engulfing(data):
            if (
                supertrend1.iloc[-1] < supertrend1.iloc[-2] and
                supertrend2.iloc[-1] < supertrend2.iloc[-2] and
                supertrend3.iloc[-1] < supertrend3.iloc[-2]
            ):
                self.signal = Signal("sell", volume)
                return self.signal

        if rsi.iloc[-1] > 70:  # RSI overbought
            self.signal = Signal("sell", volume)
            return self.signal

        if macd.iloc[-1] < signal.iloc[-1]:  # MACD bearish crossover
            self.signal = Signal("sell", volume)
            return self.signal

        if df['close'].iloc[-1] > upper_band.iloc[-1]:  # Price above Bollinger Band
            self.signal = Signal("sell", volume)
            return self.signal

        if df['close'].iloc[-1] < tenkan_sen.iloc[-1] and df['close'].iloc[-1] < kijun_sen.iloc[-1]:  # Ichimoku Cloud bearish signal
            self.signal = Signal("sell", volume)
            return self.signal

        if df['close'].iloc[-1] < sar.iloc[-1]:  # Parabolic SAR sell signal
            self.signal = Signal("sell", volume)
            return self.signal
        
        # Golden Cross / Death Cross
        if self.indicator.detect_golden_cross(short_ma, long_ma):
            self.signal = Signal("buy", volume)
            return self.signal
        if self.indicator.detect_death_cross(short_ma, long_ma):
            self.signal = Signal("sell", volume)
            return self.signal

        # RSI Divergence
        bullish_divergence, bearish_divergence = self.indicator.detect_rsi_divergence(df, rsi)
        if bullish_divergence:
            self.signal = Signal("buy", volume)
            return self.signal
        if bearish_divergence:
            self.signal = Signal("sell", volume)
            return self.signal

        # Breakout Trading
        breakout_up, breakout_down = self.indicator.detect_price_breakout(df)
        if breakout_up:
            self.signal = Signal("buy", volume)
            return self.signal
        if breakout_down:
            self.signal = Signal("sell", volume)
            return self.signal

        # Bollinger Bands Squeeze
        if self.indicator.detect_bollinger_squeeze(upper_band, lower_band):
            if breakout_up:
                self.signal = Signal("buy", volume)
                return self.signal
            if breakout_down:
                self.signal = Signal("sell", volume)
                return self.signal

        self.signal = Signal("None")
        return self.signal

    def check_exitTrade_signal(self, data):
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

        current_price = df['close'].iloc[-1]

        # Determine the volume for the exit trade
        volume = self.calculate_trade_volume()

        # Exit conditions for buy signals
        if self.signal.type == "buy":
            # Supertrend exit
            if (
                supertrend1.iloc[-1] < supertrend1.iloc[-2] or
                supertrend2.iloc[-1] < supertrend2.iloc[-2] or
                supertrend3.iloc[-1] < supertrend3.iloc[-2]
            ):
                self.signal = Signal("exit", volume)
                return self.signal

            # RSI Overbought
            if rsi.iloc[-1] > 70:
                self.signal = Signal("exit", volume)
                return self.signal

            # MACD Bearish Crossover
            if macd.iloc[-1] < signal.iloc[-1]:
                self.signal = Signal("exit", volume)
                return self.signal

            # Price above Bollinger Upper Band
            if df['close'].iloc[-1] > upper_band.iloc[-1]:
                self.signal = Signal("exit", volume)
                return self.signal

            # Ichimoku Cloud Bearish Signal
            if df['close'].iloc[-1] < tenkan_sen.iloc[-1] and df['close'].iloc[-1] < kijun_sen.iloc[-1]:
                self.signal = Signal("exit", volume)
                return self.signal

            # Parabolic SAR sell signal
            if df['close'].iloc[-1] < sar.iloc[-1]:
                self.signal = Signal("exit", volume)
                return self.signal

            # Moving Average Exit (Price crosses below short-term MA)
            if df['close'].iloc[-1] < short_ma.iloc[-1]:
                self.signal = Signal("exit", volume)
                return self.signal

            # ATR-Based Exit (Trailing Stop)
            if self.entry_price and current_price < self.entry_price - 2 * atr.iloc[-1]:
                self.signal = Signal("exit", volume)
                return self.signal

            # Fibonacci Extensions Exit (when price reaches Fibonacci extension levels)
            fib_levels = self.indicator.calculate_fibonacci_levels(df)
            if current_price >= fib_levels['100%']:  # Target reached
                self.signal = Signal("exit", volume)
                return self.signal

        # Exit conditions for sell signals
        if self.signal.type == "sell":
            # Supertrend exit
            if (
                supertrend1.iloc[-1] > supertrend1.iloc[-2] or
                supertrend2.iloc[-1] > supertrend2.iloc[-2] or
                supertrend3.iloc[-1] > supertrend3.iloc[-2]
            ):
                self.signal = Signal("exit", volume)
                return self.signal

            # RSI Oversold
            if rsi.iloc[-1] < 30:
                self.signal = Signal("exit", volume)
                return self.signal

            # MACD Bullish Crossover
            if macd.iloc[-1] > signal.iloc[-1]:
                self.signal = Signal("exit", volume)
                return self.signal

            # Price below Bollinger Lower Band
            if df['close'].iloc[-1] < lower_band.iloc[-1]:
                self.signal = Signal("exit", volume)
                return self.signal

            # Ichimoku Cloud Bullish Signal
            if df['close'].iloc[-1] > tenkan_sen.iloc[-1] and df['close'].iloc[-1] > kijun_sen.iloc[-1]:
                self.signal = Signal("exit", volume)
                return self.signal

            # Parabolic SAR buy signal
            if df['close'].iloc[-1] > sar.iloc[-1]:
                self.signal = Signal("exit", volume)
                return self.signal

            # Moving Average Exit (Price crosses above short-term MA)
            if df['close'].iloc[-1] > short_ma.iloc[-1]:
                self.signal = Signal("exit", volume)
                return self.signal

            # ATR-Based Exit (Trailing Stop)
            if self.entry_price and current_price > self.entry_price + 2 * atr.iloc[-1]:
                self.signal = Signal("exit", volume)
                return self.signal

            # Fibonacci Extensions Exit (when price retraces after hitting an extension level)
            fib_levels = self.indicator.calculate_fibonacci_levels(df)
            if current_price <= fib_levels['100%']:  # Target reached
                self.signal = Signal("exit", volume)
                return self.signal

        self.signal = Signal("None")
        return self.signal