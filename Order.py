from enum import Enum
from datetime import datetime

class ORDER_TYPE(Enum):
    BUY = 0
    SELL = 1

class Order:

    id_iterator = 0

    def __init__(self, symbol, type, volume):
        self.id = Order.id_iterator
        self.symbol = symbol
        self.type = type
        self.volume = volume
        self.open_time = datetime.now()

        Order.id_iterator += 1

    def __str__(self):
        return f"Order: {self.type.name} {self.volume} {self.symbol}"