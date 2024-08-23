import pandas as pd

class Indicator:
    def isEngulfing(self, data):
        # Previous candle (third to last)
        prev_open = data[-3]["open"]
        prev_close = data[-3]["close"]
        
        # Current candle (second to last)
        curr_open = data[-2]["open"]
        curr_close = data[-2]["close"]
        
        # Check for bullish engulfing pattern
        if prev_close < prev_open and curr_close > curr_open and curr_open < prev_close and curr_close > prev_open:
            return True
        
        # Check for bearish engulfing pattern
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

    def calculate_supertrend(self, data, period=7, multiplier=3):
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

class SignalDetector:
    def __init__(self):
        self.signal = "None"
        self.indicator = Indicator()

    def check_enterTrade_signal(self, data):
        supertrend1 = self.indicator.calculate_supertrend(data, period=10, multiplier=1)
        supertrend2 = self.indicator.calculate_supertrend(data, period=12, multiplier=3)
        supertrend3 = self.indicator.calculate_supertrend(data, period=11, multiplier=2)

        supertrend1_last = supertrend1.iloc[-1]
        supertrend1_prev = supertrend1.iloc[-2]
        supertrend2_last = supertrend2.iloc[-1]
        supertrend2_prev = supertrend2.iloc[-2]
        supertrend3_last = supertrend3.iloc[-1]
        supertrend3_prev = supertrend3.iloc[-2]

        if self.indicator.bullish_engulfing(data):
            if (
                supertrend1_last > supertrend1_prev and
                supertrend2_last > supertrend2_prev and
                supertrend3_last > supertrend3_prev):
                self.signal = "buy"
                return self.signal

        elif self.indicator.bearish_engulfing(data):
            if (
                supertrend1_last < supertrend1_prev and
                supertrend2_last < supertrend2_prev and
                supertrend3_last < supertrend3_prev):
                self.signal = "sell"
                return self.signal

        self.signal = "None"
        return self.signal
    
    def check_exitTrade_signal(self, data):
        supertrend1 = self.indicator.calculate_supertrend(data, period=10, multiplier=1)
        supertrend2 = self.indicator.calculate_supertrend(data, period=12, multiplier=3)
        supertrend3 = self.indicator.calculate_supertrend(data, period=11, multiplier=2)

        supertrend1_last = supertrend1.iloc[-1]
        supertrend1_prev = supertrend1.iloc[-2]
        supertrend2_last = supertrend2.iloc[-1]
        supertrend2_prev = supertrend2.iloc[-2]
        supertrend3_last = supertrend3.iloc[-1]
        supertrend3_prev = supertrend3.iloc[-2]

        if self.signal == "buy":
            if (
                supertrend1_last < supertrend1_prev or
                supertrend2_last < supertrend2_prev or
                supertrend3_last < supertrend3_prev):
                self.signal = "exit"
                return self.signal
        
        elif self.signal == "sell":
            if (
                supertrend1_last > supertrend1_prev or
                supertrend2_last > supertrend2_prev or
                supertrend3_last > supertrend3_prev):
                self.signal = "exit"
                return self.signal
        
        self.signal = "None"
        return self.signal