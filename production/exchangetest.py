import json
import math
import pprint
import datetime

import OkcoinRestFutureAPI
import OkcoinRestSpotAPI

pp = pprint.PrettyPrinter(indent=4, width=80)

spotapikey = '88a9e7f4-df31-455f-a5c6-0fedc23bb464'
spotsecretkey = '5111F99B0CB39B685E6834778FC4D9EB'
futapikey = 'caefd336-1862-40fd-8c3d-522bf60209d5'
futsecretkey = '59AEF7173F60A1F0BDDB01837A29359F'
okcoinspotURL = 'www.okcoin.cn'
okcoinfutuURL = 'www.okcoin.com'

symbol = "btc_usd"
contractType = "quarter"

spot = OkcoinRestSpotAPI.OKCoinSpot(okcoinspotURL, spotapikey, spotsecretkey)
future = OkcoinRestFutureAPI.OKCoinFuture(okcoinfutuURL, futapikey, futsecretkey)

# print(future.future_orderinfo(symbol=))
positions = future.future_position_4fix(symbol=symbol, contractType=contractType, type1=1)
positions = json.loads(positions)
pp.pprint(positions)
user_info = future.future_userinfo_4fix()
user_info = json.loads(user_info)
pp.pprint(user_info)
ticker = future.future_ticker(symbol=symbol, contractType=contractType)
pp.pprint(ticker)
# completely guessing on size, unnecessary? since requires 13 digits
#history = future.future_kline(symbol=symbol, type="1min", contract_type=contractType,
#                               size="100", since=str(datetime.datetime.now()))
now = int(datetime.datetime.timestamp(datetime.datetime.now()) * 1000000)
now = int(now /1000)
print(now)
now = int(datetime.datetime.timestamp(datetime.datetime.now()) * 1000000)
now = int(now /1000)
print(now)
print (datetime.datetime.timestamp(datetime.datetime.now()))
now = now
since = now - 6040000000
size = [100, 1000]
for currentsize in size:
    history = future.future_kline(symbol=symbol, type="1min", contract_type=contractType, size=str(currentsize))#,
                                  #since=str(now - currentsize))  # 6040000000 ten week? at set to
    print(len(history), history[0], history[-1])
    print("minutes",(now - history[0][0]) /60000, (now - history[-1][0])/60000, (history[-1][0] - history[0][0]) /60000)

# max contracts = (my btc * lastprice)/ticker->unit_amount all of that math.floor
# min contracts = math.ceil(maxcontracts/5.0)
max_contracts = math.floor(
    (user_info['info']['btc']['rights'] * ticker['ticker']['last'] * 20) / ticker['ticker']['unit_amount'])
min_contracts = round(max_contracts / 5.0)
if min_contracts == 0: min_contracts = 1
print(max_contracts, min_contracts)
