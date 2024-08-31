from tqdm import tqdm
from API import XTB
from config import base_url, userId, password
import json
import os

#create samples folder if it doesn't exist
if not os.path.exists("samples"):
    os.makedirs("samples")

# Create an instance of the XTB API
xtb = XTB(base_url, userId, password)

# Get all symbols
symbols = xtb.get_AllSymbols()

periods = ['M30', 'M15', 'M5']
quantity_candles = 10000
dataSample = {}
currency_errors = []
forex_currencies = [symbol['symbol'] for symbol in symbols['returnData'] if symbol['categoryName'] == 'FX']

for period in tqdm(periods, desc=f"Downloading {periods} periods"):
    for currency in tqdm(forex_currencies, desc=f"Downloading {len(forex_currencies)} forex currencies"):
        data = {}
        data["symbolInfos"] = xtb.get_Symbol(currency)
        data["symbolInfos"]['lotMinMargin'] = xtb.get_Margin(currency, data["symbolInfos"]["lotMin"])

        #using a "try catch" to avoid errors with the API and add the currency to the currency_errors list if an error occurs
        try:
            data["candles"] = xtb.get_Candles(period, currency, qty_candles=quantity_candles)[1:]
        except:
            currency_errors.append(currency)
        else:
            dataSample[currency] = data

    #add the quantity of candles to the dataSample
    dataSample["quantity_candles"] = quantity_candles

    #save data to a json file in samples folder
    with open(f"samples/forex_{period}.json", 'w') as f:
        json.dump(dataSample, f)

    #print the currencies that had an error
    print(f"Errors with the following {len(currency_errors)} currencies: {currency_errors}")

    dataSample = {}
    currency_errors = []