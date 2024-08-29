import json
from SignalDetector import SIGNAL_TYPE
from datetime import datetime

class SimulationXTB:
    def __init__(self, simulation_data_file, window_size=20):
        # Load the simulation data from a JSON file
        with open(simulation_data_file, 'r') as f:
            self.simulation_data = json.load(f)
            self.quantity_candles = self.simulation_data["quantity_candles"]
        
        # Initialize balance and other variables
        self.balance = 10000  # Assuming a starting balance for the simulation
        self.orders = {}
        self.trade_history = []
        self.window_size = window_size
        self.current_index = 0  # Index to track the sliding window position
        self.last_update_time = datetime.now()

    def get_Balance(self):
        return self.balance

    def get_AllSymbols(self):
        # print the time elapsed since the last update
        current_time = datetime.now()
        time_elapsed = current_time - self.last_update_time
        print(f"Time elapsed since last update: {time_elapsed}")
        self.last_update_time = current_time
        # Slide the window by one
        self.current_index += 1
        if self.current_index + self.window_size > self.quantity_candles:
            return None  # Indicating the end of the data
        
        #print current index every 5% of the data
        if self.current_index % (self.quantity_candles // 20) == 0:
            print(f"Current index: {self.current_index}")
        # Extract and return the symbol info from the simulation data
        symbols = list(self.simulation_data.keys())
        symbols.pop()
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
                'required_margin': required_margin  # Store the margin used
            }

        elif transaction_type == SIGNAL_TYPE.CLOSE:
            if order in self.orders:
                original_order = self.orders[order]
                entry_price = original_order['entry_price']
                initial_margin = original_order['required_margin']
                
                # Calculate profit or loss
                if original_order['cmd'] == "buy":  # Closing a buy order
                    profit_or_loss = (price - entry_price) * volume
                elif original_order['cmd'] == "sell":  # Closing a sell order
                    profit_or_loss = (entry_price - price) * volume
                
                # Update balance based on the profit or loss and refund the initial margin
                self.balance += profit_or_loss + initial_margin

                # Remove the closed order
                del self.orders[order]
        
        self.trade_history.append({
            'symbol': symbol,
            'cmd': cmd,
            'transaction_type': transaction_type,
            'volume': volume,
            'order': order,
            'price': price,
            'balance_after_trade': self.balance
        })

    def get_performance_metrics(self):
        total_trades = len(self.trade_history)
        winning_trades = len([trade for trade in self.trade_history if trade['balance_after_trade'] > self.balance])
        losing_trades = total_trades - winning_trades
        final_balance = self.balance
        profit_or_loss = final_balance - 10000  # Assuming starting balance was 10000
        
        metrics = {
            "Total Trades": total_trades,
            "Winning Trades": winning_trades,
            "Losing Trades": losing_trades,
            "Final Balance": final_balance,
            "Profit/Loss": profit_or_loss
        }
        
        return metrics