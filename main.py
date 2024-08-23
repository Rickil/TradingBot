from config import base_url, userId, password
from API import XTB

xstore = XTB(base_url, userId, password)
print(xstore.ping())
xstore.disconnect()