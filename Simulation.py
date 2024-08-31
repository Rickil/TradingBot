import json
from SignalDetector import SIGNAL_TYPE
from Order import ORDER_TYPE
from datetime import datetime

class SimulationXTB:
    def __init__(self, simulation_data_file, window_size=20, sample_size=500):
        # Load the simulation data from a JSON file
        with open(simulation_data_file, 'r') as f:
            self.simulation_data = json.load(f)
            self.quantity_candles = self.simulation_data["quantity_candles"]
        
        # Initialize balance and other variables
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.orders = {}
        self.window_size = window_size
        self.current_index = 0  # Index to track the sliding window position
        self.sample_size = min(sample_size, self.quantity_candles)

    def get_Balance(self):
        return self.balance
    
    def updateSimulationState(self):
        # Slide the window by one
        self.current_index += 1



    def get_AllSymbols(self):
        self.updateSimulationState()

        if self.current_index + self.window_size > self.sample_size:
            return None  # Indicating the end of the data
        
        #print the progression every 5% of the data
        if self.current_index % (self.sample_size // 20) == 0:
            print(f"Progress: {self.current_index / self.sample_size * 100:.2f}%")
            
        # Extract and return the symbol info from the simulation data
        symbols = list(self.simulation_data.keys())
        symbols.pop()
        symbols = symbols[:20]  # Limit to the first 2 symbols for demonstration
        all_symbols_info = [self.simulation_data[symbol]['symbolInfos'] for symbol in symbols]
        return {'returnData': all_symbols_info}

    def get_Candles(self, period, symbol, qty_candles):
        # Return the current window of candle data for the specified symbol
        start_index = self.current_index
        end_index = start_index + self.window_size
        
        # Ensure the window does not exceed the available data
        if end_index > len(self.simulation_data[symbol]['candles']):
            end_index = len(self.simulation_data[symbol]['candles'])

        return self.simulation_data[symbol]['candles'][start_index:end_index]

    def get_Margin(self, symbol, volume):
        # Get the symbol info from the simulation data
        symbol_info = self.simulation_data[symbol]['symbolInfos']
        lot_min = symbol_info['lotMin']
        lot_min_margin = symbol_info['lotMinMargin']
        
        # Calculate the required margin based on the given volume
        required_margin = (volume / lot_min) * lot_min_margin
        return required_margin

    def make_Trade(self, symbol, cmd, transaction_type, volume, order=None):
        # Simulate making a trade by updating balance and storing the order
        last_candle = self.simulation_data[symbol]['candles'][self.current_index + self.window_size - 1]
        price = last_candle['open']
        
        if transaction_type == SIGNAL_TYPE.OPEN:
            # Calculate the required margin using the provided margin calculation method
            required_margin = self.get_Margin(symbol, volume)
            self.balance -= required_margin
            
            # Create a new order with entry price
            order_id = len(self.orders) + 1
            self.orders[order_id] = {
                'symbol': symbol,
                'volume': volume,
                'cmd': cmd,
                'entry_price': price,
                'required_margin': required_margin,  # Store the margin used
                'success': False
            }

        elif transaction_type == SIGNAL_TYPE.CLOSE:
            if order in self.orders:
                price = self.simulation_data[self.orders[order]['symbol']]['candles'][self.current_index + self.window_size - 1]['open']
                original_order = self.orders[order]
                entry_price = original_order['entry_price']
                initial_margin = original_order['required_margin']
                
                # Calculate profit or loss
                profit_or_loss = 0
                if original_order['cmd'] == ORDER_TYPE.BUY:  # Closing a buy order
                    profit_or_loss = (price - entry_price)
                elif original_order['cmd'] == ORDER_TYPE.SELL:  # Closing a sell order
                    profit_or_loss = (entry_price - price)
                
                # Update balance based on the profit or loss and refund the initial margin
                self.balance += profit_or_loss * initial_margin + initial_margin

                #print "window_size" last prices of candles on open
                #print(f"{symbol} window_size last prices of candles on open: {self.simulation_data[symbol]['candles'][self.current_index + self.window_size - 1]}")

                #print price and entry price
                #print(f"Price: {price}, Entry Price: {entry_price}")
                #print(f"Order {order} closed with a profit/loss of {profit_or_loss}")

                if profit_or_loss > 0:
                    self.orders[order]["success"] = True

    def get_performance_metrics(self):
        total_trades = len(list(self.orders.keys()))
        winning_trades = sum([1 for order in self.orders.values() if order["success"]])
        losing_trades = total_trades - winning_trades
        final_balance = self.balance
        profit_or_loss = final_balance - self.initial_balance
        
        metrics = {
            "Total Trades": total_trades,
            "Winning Trades": winning_trades,
            "Losing Trades": losing_trades,
            "Final Balance": final_balance,
            "Profit/Loss": profit_or_loss
        }
        
        return metrics