from SignalDetector import SignalDetector
from Order import Order
from API import XTB, SimulationXTB

class Bot:

    def __init__(self, name, url, userId, password, period="H1", qty_candles=500, simulation=False, simulation_data_file=None):
        self.name = name
        self.period = period
        self.qty_candles = qty_candles
        self.simulation = simulation

        if self.simulation:
            # Initialize with SimulationXTB if simulation mode is enabled
            if simulation_data_file is None:
                raise ValueError("simulation_data_file must be provided when simulation mode is enabled")
            self.xtb = SimulationXTB(simulation_data_file)
        else:
            # Initialize with real XTB API if not in simulation mode
            self.xtb = XTB(url, userId, password)

        # Initialize balance
        self.balance = self.xtb.get_Balance()

        # Initialize orders
        self.orders = {}

        # Initialize signal detector
        self.signal_detector = SignalDetector()

        # Initialize symbol infos
        self.symbolInfos = {}

        # Initialize data
        self.data = {}

        self.allSymbolsInfos = self.xtb.get_AllSymbols()['returnData']
        for symbolInfos in self.allSymbolsInfos:
            symbol = symbolInfos['symbol']
            lotMinMargin = self.xtb.get_Margin(symbol, symbolInfos['lotMin'])
            symbolInfos['lotMinMargin'] = lotMinMargin
            symbol_data = self.xtb.get_Candles(period=self.period, symbol=symbol, qty_candles=self.qty_candles)[1:]
            self.data[symbol] = symbol_data
            self.symbolInfos[symbol] = symbolInfos
            self.orders[symbol] = []

        print(f"Bot {self.name} is ready")

    def updateData(self):
        self.allSymbolsInfos = self.xtb.get_AllSymbols()
        for symbolInfos in self.allSymbolsInfos:
            symbol = symbolInfos['symbol']
            lotMinMargin = self.xtb.get_Margin(symbol, symbolInfos['lotMin'])
            symbolInfos['lotMinMargin'] = lotMinMargin
            symbol_data = self.xtb.get_Candles(period=self.period, symbol=symbol, qty_candles=self.qty_candles)[1:]
            self.data[symbol] = symbol_data
            self.symbolInfos[symbol] = symbolInfos
    
    def run(self):
        while True:
            self.updateData()
            for symbol in list(self.symbolInfos.keys()):
                # Check if we have to close orders
                for order in self.orders[symbol]:
                    signal = self.signal_detector.check_closeTrade_signal(self.data[symbol], order)
                    if signal:
                        self.xtb.make_Trade(symbol=symbol, cmd=signal.cmd, transaction_type=signal.type, volume=signal.volume, order=order.id)
                        self.orders[symbol].remove(order)
            
            # Update balance after closed trades
            self.balance = self.xtb.get_Balance()

            for symbol in list(self.symbolInfos.keys()):
                # Check if we have to open orders
                signal = self.signal_detector.check_enterTrade_signal(self.data[symbol], self.balance, self.symbolInfos[symbol])
                if signal:
                    order = Order(symbol=symbol, type=signal.type, volume=signal.volume)
                    self.orders[symbol].append(order)
                    self.xtb.make_Trade(symbol=symbol, cmd=signal.cmd, transaction_type=signal.type, volume=signal.volume, order=order.id)
                    # Update balance after a trade is made
                    self.balance = self.xtb.get_Balance()
