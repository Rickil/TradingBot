from SignalDetector import SignalDetector, SIGNAL_TYPE
from Order import Order
from API import XTB

class Bot:

    def __init__(self, name, url, userId, password, period="H1", qty_candles=500):
        self.name = name
        self.period = period
        self.qty_candles = qty_candles

        #init connection with the server
        self.xtb = XTB(url, userId, password)

        #init balance
        self.balance = self.xtb.get_Balance()

        #init orders
        self.orders = {}

        #init signal detector
        self.signal_detector = SignalDetector()

        #init symbol infos
        self.symbolInfos = {}

        #init data
        self.data = {}

        allSymbolsInfos = self.xtb.get_AllSymbols()['returnData']
        for symbolInfos in allSymbolsInfos:
            symbol = symbolInfos['symbol']
            symbol_data = self.xtb.get_Candles(period=self.period, symbol=symbol, qty_candles=self.qty_candles)[1:]
            self.data[symbol] = symbol_data
            self.symbolInfos[symbol] = symbolInfos
            self.orders[symbol] = []

        print(f"Bot {self.name} is ready")

    def updateData(self):
        allSymbolsInfos = self.xtb.get_AllSymbols()
        for symbolInfos in allSymbolsInfos:
            symbol = symbolInfos['symbol']
            symbol_data = self.xtb.get_Candles(period=self.period, symbol=symbol, qty_candles=self.qty_candles)[1:]
            self.data[symbol] = symbol_data
            self.symbolInfos[symbol] = symbolInfos
    
    def run(self):
        while True:
            self.updateData()
            for symbol in list(self.symbolInfos.keys()):
                #check if we have to close orders
                for order in self.orders[symbol]:
                    signal = self.signal_detector.check_closeTrade_signal(self.data[symbol], order)
                    if signal:
                        self.xtb.make_Trade(symbol=symbol, cmd=signal.cmd, transaction_type=signal.type, volume=signal.volume, order=order.id)
                        self.orders[symbol].remove(order)
            
            #update balance after closed trades
            self.balance = self.xtb.get_Balance()

            for symbol in list(self.symbolInfos.keys()):
                #check if we have to open orders
                signal = self.signal_detector.check_enterTrade_signal(self.data[symbol], self.balance, self.symbolInfos[symbol])
                if signal:
                    order = Order(symbol=symbol, type=signal.type, volume=signal.volume)
                    self.orders[symbol].append(order)
                    self.xtb.make_Trade(symbol=symbol, cmd=signal.cmd, transaction_type=signal.type, volume=signal.volume, order=order.id)
                    #update balance after a trade is made
                    self.balance = self.xtb.get_Balance()
