import hashlib
import sys
import zlib
from datetime import timedelta, datetime
import json
import codecs
import pprint as pp

import websocket

api_key = ''
secret_key = ""


# business
def buildMySign(params, secretKey):
    sign = ''
    for key in sorted(params.keys()):
        sign += key + '=' + str(params[key]) + '&'
    return hashlib.md5((sign + 'secret_key=' + secretKey).encode("utf-8")).hexdigest().upper()


# spot trade
def spotTrade(channel, api_key, secretkey, symbol, tradeType, price='', amount=''):
    params = {
        'api_key': api_key,
        'symbol': symbol,
        'type': tradeType
    }
    if price:
        params['price'] = price
    if amount:
        params['amount'] = amount
    sign = buildMySign(params, secretkey)
    finalStr = "{'event':'addChannel','channel':'" + channel + "','parameters':{'api_key':'" + api_key + "',\
                'sign':'" + sign + "','symbol':'" + symbol + "','type':'" + tradeType + "'"
    if price:
        finalStr += ",'price':'" + price + "'"
    if amount:
        finalStr += ",'amount':'" + amount + "'"
    finalStr += "},'binary':'true'}"
    return finalStr


# spot cancel order
def spotCancelOrder(channel, api_key, secretkey, symbol, orderId):
    params = {
        'api_key': api_key,
        'symbol': symbol,
        'order_id': orderId
    }
    sign = buildMySign(params, secretkey)
    return "{'event':'addChannel','channel':'" + channel + "','parameters':{'api_key':'" + api_key + "','sign':'" + sign + "','symbol':'" + symbol + "','order_id':'" + orderId + "'},'binary':'true'}"


# subscribe trades for self
def realtrades(channel, api_key, secretkey):
    params = {'api_key': api_key}
    sign = buildMySign(params, secretkey)
    return "{'event':'addChannel','channel':'" + channel + "','parameters':{'api_key':'" + api_key + "','sign':'" + sign + "'},'binary':'true'}"


# trade for future
def future_trade(api_key, secretkey, symbol, contractType, price='', amount='', tradeType='', matchPrice='',
                leverRate=''):
    params = {
        'api_key': api_key,
        'symbol': symbol,
        'contract_type': contractType,
        'amount': amount,
        'type': tradeType,
        'match_price': matchPrice,
        'lever_rate': leverRate
    }
    if price:
        params['price'] = price
    sign = buildMySign(params, secretkey)
    finalStr = "{'event':'addChannel','channel':'ok_futuresusd_trade','parameters':{'api_key':'" + api_key + "',\
               'sign':'" + sign + "','symbol':'" + symbol + "','contract_type':'" + contractType + "'"
    if price:
        finalStr += ",'price':'" + price + "'"
    finalStr += ",'amount':'" + amount + "','type':'" + tradeType + "','match_price':'" + matchPrice + "','lever_rate':'" + leverRate + "'},'binary':'true'}"
    return finalStr


# future trade cancel
def futureCancelOrder(api_key, secretkey, symbol, orderId, contractType):
    params = {
        'api_key': api_key,
        'symbol': symbol,
        'order_id': orderId,
        'contract_type': contractType
    }
    sign = buildMySign(params, secretkey)
    return "{'event':'addChannel','channel':'ok_futuresusd_cancel_order','parameters':{'api_key':'" + api_key + "',\
            'sign':'" + sign + "','symbol':'" + symbol + "','contract_type':'" + contractType + "','order_id':'" + orderId + "'},'binary':'true'}"


# subscribe future trades for self
def futureRealTrades(api_key, secretkey):
    params = {'api_key': api_key}
    sign = buildMySign(params, secretkey)
    return "{'event':'addChannel','channel':'ok_sub_futureusd_trades','parameters':{'api_key':'" + api_key + "','sign':'" + sign + "'},'binary':'true'}"


def on_open(self):
    futapikey = 'f042eed5-faa0-4f14-98d8-045d2c5acc4d'
    futsecretkey = '3D0432CD2D798DFA8D91CEEBC4C17F88'
    params = {'api_key': futapikey}
    sign = buildMySign(params, futsecretkey)
    # subscribe okcoin.com spot ticker
    # self.send("{'event':'addChannel','channel':'ok_sub_spotusd_btc_ticker','binary':'true'}")

    # subscribe okcoin.com future this_week ticker
    # self.send("{'event':'addChannel','channel':'ok_sub_futureusd_btc_ticker_quarter'}")

    # subscribe okcoin.com future depth
    # self.send("{'event':'addChannel','channel':'ok_sub_futureusd_btc_depth_quarter_20','binary':'true'}")

    # subscrib real trades for self
    #realtradesMsg = realtrades('ok_sub_spotusd_trades',api_key,secret_key)
    #self.send(realtradesMsg)

    self.send("{'event':'addChannel','channel':'ok_sub_futureusd_userinfo','parameters':{'api_key':'" + futapikey + "','sign':'" + sign + "'},'binary':'true'}")
    self.send("{'event':'addChannel','channel':'ok_sub_futureusd_positions','parameters':{'api_key':'" + futapikey + "','sign':'" + sign + "'},'binary':'true'}")

    # spot trade via websocket
    # spotTradeMsg = spotTrade('ok_spotusd_trade',api_key,secret_key,'ltc_usd','buy_market','1','')
    # self.send(spotTradeMsg)


    # spot trade cancel
    # spotCancelOrderMsg = spotCancelOrder('ok_spotusd_cancel_order',api_key,secret_key,'btc_usd','125433027')
    # self.send(spotCancelOrderMsg)

    # future trade
    #futureTradeMsg = future_trade(api_key,secret_key,'btc_usd','this_week','','2','1','1','20')
    #self.send(futureTradeMsg)

    # future trade cancel
    # futureCancelOrderMsg = futureCancelOrder(api_key,secret_key,'btc_usd','65464','this_week')
    # self.send(futureCancelOrderMsg)

    # subscrbe future trades for self
    #futureRealTradesMsg = futureRealTrades(api_key,secret_key)
    #self.send(futureRealTradesMsg)
    self.send(
       "{'event':'addChannel','channel':'ok_sub_futureusd_trades','parameters':{'api_key':'" + futapikey + "','sign':'" + sign + "'},'binary':'true'}")
    # self.send(
    #     "{'event':'addChannel','channel':'ok_sub_futureusd_userinfo','parameters':{'api_key':'" + futapikey + "','sign':'" + sign + "'},'binary':'true'}")
    # # sub self positions -> current long & short
    self.send(
        "{'event':'addChannel','channel':'ok_sub_futureusd_positions','parameters':{'api_key':'" + futapikey + "','sign':'" + sign + "'},'binary':'true'}")
    # # sub self trades -> my current trades in books and historical trades.
    # self.send(
    #     "{'event':'addChannel','channel':'ok_sub_futureusd_trades','parameters':{'api_key':'" + futapikey + "','sign':'" + sign + "'},'binary':'true'}")
    self.send(
        "{'event':'login','channel':'login','parameters':{'api_key':'" + futapikey + "','sign':'" + sign + "'},'binary':'1'}")

def on_message(self, evt):
    try:
        data = inflate(evt)  # data decompress
        data = data.decode('utf-8')
    except TypeError as e:
        #print(e)
        data = str(evt)
    data = json.loads(data)

    #pp.pprint(data)
    for tickdata in data:
        pp.pprint(tickdata)


def inflate(data):
    decompress = zlib.decompressobj(
        -zlib.MAX_WBITS  # see above
    )
    inflated = decompress.decompress(data)
    inflated += decompress.flush()
    return inflated


def on_error(self, evt):
    print(evt)


def on_close(self):
    print('DISCONNECT')


if __name__ == "__main__":
    url = "wss://real.okcoin.com:10440/websocket/okcoinapi"  # if okcoin.cn  change url wss://real.okcoin.cn:10440/websocket/okcoinapi
    api_key = 'your api_key which you apply'
    secret_key = "your secret_key which you apply"

    websocket.enableTrace(False)
    if len(sys.argv) < 2:
        host = url
    else:
        host = sys.argv[1]
    lastmessagetime = datetime.now()
    while True:
        try:
            ws = websocket.WebSocketApp(host,
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close)
            ws.on_open = on_open
            ws.run_forever(ping_interval=6, ping_timeout=3)
        except KeyboardInterrupt:
            break
        except TimeoutError:
            continue
        else:
            break

