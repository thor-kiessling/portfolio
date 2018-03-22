import copy
# import fnmatch
import hashlib
import json
import logging
import requests
# import math
# import os
import threading
from multiprocessing import Process, Manager, Pool
# import multiprocessing
# import multiprocessing.pool
import talib.stream as ta
import pickle
import os
import math
import sys

import time
import zlib
from datetime import datetime, timedelta
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# from memory_profiler import profile
# from matplotlib import pyplot as plt
# import collections
# from subprocess import run, DEVNULL, PIPE
import apsw

import globalvar
import numpy as np
import websocket
#from sklearn.externals import joblib

#import build_ta_data
import get_history

# malleable globals ##########################

logging.basicConfig(level=logging.DEBUG)


def inflate(data):
    decompress = zlib.decompressobj(
        -zlib.MAX_WBITS  # see above
    )
    inflated = decompress.decompress(data)
    inflated += decompress.flush()
    return inflated


def buildMySign(params, secretKey):
    sign = ''
    for key in sorted(params.keys()):
        sign += key + '=' + str(params[key]) + '&'
    return hashlib.md5((sign + 'secret_key=' + secretKey).encode("utf-8")).hexdigest().upper()


def fut_on_open(self):
    futapikey = 'f042eed5-faa0-4f14-98d8-045d2c5acc4d'
    futsecretkey = '3D0432CD2D798DFA8D91CEEBC4C17F88'
    params = {'api_key': futapikey}
    sign = buildMySign(params, futsecretkey)
    # subscribe okcoin.com futureusd ticker -> know many contracts can buy, know when to withdraw
    self.send("{'event':'addChannel','channel':'ok_sub_futureusd_btc_ticker_quarter'}")
    # sub orderbook depth -> know where to place trades
    self.send("{'event':'addChannel','channel':'ok_sub_future_btc_depth_quarter_20'}")
    # sub self info -> btc balance
    self.send(
        "{'event':'login','channel':'login','parameters':{'api_key':'" + futapikey + "','sign':'" + sign + "'}}")
    # self.send(
    #     "{'event':'addChannel','channel':'ok_sub_futureusd_userinfo','parameters':{'api_key':'" + futapikey + "','sign':'" + sign + "'}}")
    # # sub self positions -> current long & short
    # self.send(
    #     "{'event':'addChannel','channel':'ok_sub_futureusd_positions','parameters':{'api_key':'" + futapikey + "','sign':'" + sign + "'}}")
    # # sub self trades -> my current trades in books and historical trades.
    # self.send(
    #     "{'event':'addChannel','channel':'ok_sub_futureusd_trades','parameters':{'api_key':'" + futapikey + "','sign':'" + sign + "'}}")

def on_ping(self):
    while glob_futws.sock is None or not glob_futws.sock.connected:
        time.sleep(1)
    pass

def spot_on_message(self, evt):
    data = inflate(evt)  # data decompress
    data = data.decode('utf-8')
    data = json.loads(data)
    for tickdata in data:
        if 'success' in tickdata:
            if tickdata['success'] == 'true':
                print("spot ticker established successfully")
            else:
                raise TimeoutError
            return

        if not 'data' in tickdata: return
        dateobj = datetime.utcfromtimestamp(float(str(tickdata['data']['timestamp'])[0:-3]))  # timestamp includes ms
        tickdata['data']['time'] = dateobj.strftime("%Y-%m-%dT%H:%M:%SZ")
        globalvar.l.acquire()
        # print(tickdata)
        try:
            globalvar.dbcur.execute("INSERT OR IGNORE INTO spot VALUES (?,?,?,?,?)",
                                    [tickdata['data']['time'], tickdata['data']['last'], tickdata['data']['high'],
                                     tickdata['data']['low'], tickdata['data']['vol']])
        finally:
            globalvar.l.release()
        return


def fut_on_message(self, evt):  # should not ever block, will be many many calls
    try:
        data = inflate(evt)  # data decompress
        data = data.decode('utf-8')
    except TypeError as e:
        #print(e)
        data = str(evt)
    data = json.loads(data)
    #data = data[0]
    if isinstance(data, dict):
        print("got error", data)
        raise BaseException

    try:
        for msg in data:
            #print(msg)
            #msg = dict(msg)
            if msg['channel'] == 'addChannel':
                if msg['data']['result']:
                    print(msg['data']['channel'] + " established successfully")
                    self.namespace.error_count = 0
                    return
                else:
                    print(msg['data']['channel'] + " NOT established successfully")
                    return
            if msg['channel'] == 'login':
                if msg['data']['result']:
                    print("logged into the exchange")
                    return
                else:
                    print("error on trying to login to the exchange")
                    return

            if 'success' in msg:
                if msg['result'] == 'True':
                    print(msg['channel'] + " established successfully") #todo this is the old format messaging for established channel
                    self.namespace.error_count = 0
                    return
                else:
                    self.namespace.error_count += 1

                    if msg['channel'] == 'ok_futureusd_cancel_order' and msg['success'] == 'false':  # on cancel failure
                         # if max_contracts = 0, have it send 1 optimistic long and one optimistic short. bootstrap
                         # plot slopes + points of the crazy lines to debug them. consider a linear fit on fewer points
                        print(msg)
                        #glob_futws.send(future_get_orders(ns, futapikey, futsecretkey, symbol, contractType))

                         # tradestemp = copy.deepcopy(self.namespace.glob_trades)
                        # msg = copy.deepcopy(msg)
                        # tradestemp.pop(int(msg['order_id']), None)
                        # self.namespace.glob_trades = tradestemp
                        #self.namespace.glob_trades = {}  # clear out my understanding of my open trades because ghost orders
                        # consider reinstating ^, then sending 1 long+short.
                    else:
                        print(msg)
            if 'data' in msg:
                if 'result' in msg['data']:
                    if msg['data']['result'] == False:
                        if msg['channel'] == 'ok_futureusd_cancel_order' and msg['data']['error_code'] == 20015:
                            #self.namespace.glob_trades = {}
                            self.namespace.ghost_trade_counter += 1
                            return
                        if msg['channel'] == 'ok_futuresusd_trade' and msg['data']['error_code'] == 20007:
                            self.namespace.quarter_exists = False
                            return
                        if msg['channel'] == 'ok_futureusd_cancel_order' and msg['data']['error_code'] == 20049:
                            self.namespace.glob_trades.clear()
                        notification("GotErrorFromExchange", str(msg))
                        print(msg)
                        return
            # if msg['channel'] != 'ok_sub_futureusd_btc_depth_quarter_20' and msg['channel'] != 'ok_sub_futureusd_btc_ticker_quarter':
            #    print(msg)
            if str(msg['channel']) == 'ok_sub_futureusd_userinfo':
                # do userinfo stuff
                tempvar = 0.0
                tempvar = float(msg['data']['balance'])
                #for contract in msg['data']['contracts']:
                    #tempvar += float(contract['bond'])  # this was for fixed margin
                self.namespace.contract_balance = float(msg['data']['keep_deposit'])
                tempvar += float(msg['data']['profit_real'])
                self.namespace.glob_btc_balance = tempvar
                print("account balance:", tempvar, "$", tempvar * ns.glob_future_price)  # todo remove temp
            elif str(msg['channel']) == 'ok_sub_futureusd_btc_ticker_quarter':
                msg = copy.deepcopy(msg)
                # do ticker stuff
                tempticker = float(msg['data']['last'])
                self.namespace.glob_future_price = tempticker
                self.namespace.glob_last_tick = datetime.utcnow()
                # print("                got a ticker update at ", datetime.now())
            elif str(msg['channel']) == 'ok_sub_future_btc_depth_quarter_20':
                # do marketdepth
                msg = copy.deepcopy(msg)
                # trade_proc = threading.Thread(target=desired_position, args=(self, msg['data']), name="trade_subproc")
                tempbook = msg['data']
                #self.namespace.glob_orderbook.clear()
                self.namespace.glob_orderbook.update(tempbook)
                self.namespace.glob_last_ob = datetime.utcnow()
            elif str(msg['channel']) == 'ok_sub_futureusd_positions':
                # get current positions
                msg = copy.deepcopy(msg)
                for positions in msg['data']['positions']:
                    if int(positions['position']) == 1:#int(positions['lever_rate']) == 20 and
                        self.namespace.glob_current_longs = int(float(positions['hold_amount']))
                    if int(positions['position']) == 2:#int(positions['lever_rate']) == 20 and
                        self.namespace.glob_current_shorts = int(float(positions['hold_amount']))
            elif str(msg['channel']) == 'ok_sub_futureusd_trades' :  # keep only active trades.
                # store trades in dict by key 'order_id'
                #tradestemp = copy.deepcopy(self.namespace.glob_trades)
                msg = copy.deepcopy(msg)
                if int(msg['data']['status']) == 0 or int(msg['data']['status']) == 1:  # order is active on books
                    if 'orderid' in msg['data']:
                        msg['data']['order_id'] = msg['data']['orderid']

                        #self.namespace.glob_trades[int(msg['data']['orderid'])] = dict(msg['data'])  # access glob_trades['99'] will give dict
                    if msg['data']['type'] == 1:
                        self.namespace.book_contracts_open_long += int(
                            msg['data']['amount'] - int(msg['data']['deal_amount']))
                    elif msg['data']['type'] == 2:
                        self.namespace.book_contracts_open_short += int(
                            msg['data']['amount'] - int(msg['data']['deal_amount']))
                    elif msg['data']['type'] == 3:
                        self.namespace.book_contracts_close_long += int(
                            msg['data']['amount'] - int(msg['data']['deal_amount']))
                    elif msg['data']['type'] == 4:
                        self.namespace.book_contracts_close_short += int(
                            msg['data']['amount'] - int(msg['data']['deal_amount']))
                    self.namespace.glob_trades[int(msg['data']['order_id'])] = dict(msg['data'])  # access glob_trades['99'] will give dict
                    #self.namespace.glob_trades = tradestemp
                if int(msg['data']['status']) == -1 or int(msg['data']['status']) == 2:  # order is no longer active
                    if 'orderid' in msg['data']:
                        msg['data']['order_id'] = msg['data']['orderid']

                    self.namespace.glob_trades.pop(int(msg['data']['order_id']), None)

                    if msg['data']['type'] == 1:
                        self.namespace.book_contracts_open_long -= int(msg['data']['amount'])
                    elif msg['data']['type'] == 2:
                        self.namespace.book_contracts_open_short -= int(msg['data']['amount'])
                    elif msg['data']['type'] == 3:
                        self.namespace.book_contracts_close_long -= int(msg['data']['amount'])
                    elif msg['data']['type'] == 4:
                        self.namespace.book_contracts_close_short -= int(msg['data']['amount'])

                    #self.namespace.glob_trades = tradestemp

                    # self.namespace.glob_trades = {}
                    # eliminate the key from the dict
            elif str(msg['channel']) == 'ok_futureusd_orderinfo':
                # store trades in dict by key 'order_id'
                # tradestemp = copy.deepcopy(self.namespace.glob_trades)
                # ignore status 4 because in-progress cancel
                msg = copy.deepcopy(msg)
                for open_order in msg['data']['orders']:
                    if int(open_order['status']) == 0 or int(open_order['status']) == 1:  # order is active on books
                        self.namespace.glob_trades[int(open_order['order_id'])] = dict(open_order)  # access glob_trades['99'] will give dict
                        if open_order['type'] == 1:
                            self.namespace.book_contracts_open_long += int(open_order['amount'] - int(open_order['deal_amount']))
                        elif open_order['type'] == 2:
                            self.namespace.book_contracts_open_short += int(open_order['amount'] - int(open_order['deal_amount']))
                        elif open_order['type'] == 3:
                            self.namespace.book_contracts_close_long += int(open_order['amount'] - int(open_order['deal_amount']))
                        elif open_order['type'] == 4:
                            self.namespace.book_contracts_close_short += int(open_order['amount'] - int(open_order['deal_amount']))
                        # self.namespace.glob_trades = tradestemp
                    if int(open_order['status']) == -1 or int(open_order['status']) == 2:  # order is no longer active
                        self.namespace.glob_trades.pop(int(open_order['order_id']), None)
                        if open_order['type'] == 1:
                            self.namespace.book_contracts_open_long -= int(open_order['amount'])
                        elif open_order['type'] == 2:
                            self.namespace.book_contracts_open_short -= int(open_order['amount'])
                        elif open_order['type'] == 3:
                            self.namespace.book_contracts_close_long -= int(open_order['amount'])
                        elif open_order['type'] == 4:
                            self.namespace.book_contracts_close_short -= int(open_order['amount'])
                        #self.namespace.book_contracts -= int(open_order['amount'])
                #self.namespace.book_contracts = 0
    except BaseException as e:
        notification("BotGotError", str(e) + str(data))
        print("got error in future message " + str(e) + str(data))

# def keepdbupdated(ns):
#     url = "wss://real.okcoin.cn:10440/websocket/okcoinapi"  # if okcoin.cn  change url wss://real.okcoin.cn:10440/websocket/okcoinapi
#     while True:
#         try:
#             spotws = websocket.WebSocketApp(url,
#                                             on_message=spot_on_message,
#                                             on_error=on_error,
#                                             on_close=on_close, namespace=ns)
#             spotws.on_open = spot_on_open
#             spotws.run_forever(ping_interval=6, ping_timeout=3)
#
#         except KeyboardInterrupt:
#             break
#         except Exception as e:
#             continue
#         else:
#             continue


def fill_gap(ns):
    while True:
        try:
            time.sleep(7)
            get_history.main(ns)
        except:
            time.sleep(20)


def future_cancel_order(ns, api_key, secretkey, symbol, orderId, contractType):

    params = {
        'api_key': api_key,
        'contract_type': contractType,
        'order_id': orderId,
        'symbol': symbol,
    }
    sign = buildMySign(params, secretkey)
    while glob_futws.sock is None or not glob_futws.sock.connected:
        time.sleep(1)
    return "{'event':'addChannel','channel':'ok_futureusd_cancel_order','parameters':{'api_key':'" + api_key + "',\
            'sign':'" + sign + "','symbol':'" + symbol + "','order_id':'" + orderId + "','contract_type':'" + contractType +"'}}"


def future_get_orders(ns, api_key, secretkey, symbol, contractType, orderId=None):
    if not orderId:
        orderId = "-1"
        status = "1"
    else:
        status = "2"
    current_page = "1"
    page_length = "15"
    params = {
        'api_key': api_key,
        'contract_type': contractType,
        'current_page': current_page,
        'order_id': orderId,
        'page_length': page_length,
        'status': status,
        'symbol': symbol,
    }
    sign = buildMySign(params, secretkey)
    finalStr = "{'event':'addChannel','channel':'ok_futureusd_orderinfo','parameters':{'api_key':'" + api_key + "','sign':'" + sign + "','symbol':'" + symbol + "','order_id':'" + orderId + "','contract_type':'" + contractType + "','status':'" + status + "','current_page':'" + current_page + "','page_length':'" + page_length + "'}}"
    while glob_futws.sock is None or not glob_futws.sock.connected:
        time.sleep(1)
    return finalStr


def future_trade(ns, api_key, secretkey, symbol, contractType, price='', amount='', tradeType='', matchPrice='',
                 leverRate=''):
    params = {
        'amount': amount,
        'api_key': api_key,
        'contract_type': contractType,
        'lever_rate': leverRate,
        'match_price': matchPrice,
        'symbol': symbol,
        'type': tradeType,
    }

    if price:
        params['price'] = price
    sign = buildMySign(params, secretkey)
    finalStr = "{'event':'addChannel','channel':'ok_futuresusd_trade','parameters':{'api_key':'" + api_key + "',\
               'sign':'" + sign + "','symbol':'" + symbol + "','contract_type':'" + contractType + "'"
    if price:
        finalStr += ",'price':'" + price + "'"
    finalStr += ",'amount':'" + amount + "','type':'" + tradeType + "','match_price':'" + matchPrice + "','lever_rate':'" + leverRate + "'}}"
    while glob_futws.sock is None or not glob_futws.sock.connected:
        time.sleep(1)
    return finalStr


def close_all(future, positions):
    symbol = "btc_usd"
    contractType = "quarter"
    if not positions['holding']:
        current_longs, current_shorts = 0, 0
    else:
        current_longs = positions['holding'][0]['buy_available']
        current_shorts = positions['holding'][0]['sell_available']
    if current_shorts != 0:
        # close shorts
        future.future_trade(symbol=symbol, contractType=contractType, amount=current_shorts,
                            tradeType=4, matchPrice=1, leverRate=20)
        print("liquidating shorts")
    if current_longs != 0:
        # close longs
        future.future_trade(symbol=symbol, contractType=contractType, amount=current_longs,
                            tradeType=3, matchPrice=1, leverRate=20)
        print("liquidating longs")

def getkey(item):
    return item[0]

def desired_position(ns):
    # ns.reversal_flag = False
    if ns.entry_price == 0:
        ns.entry_price = ns.glob_future_price
        pickle.dump(ns.entry_price, open('entry_price', 'wb'))
    # ns.last_trade_time = False
    maximum_loss = -.02  # todo fill in with real/trained value currently .5%
    reentry_percent = -1.0 * maximum_loss  # todo fill in with real/trained value
    if ns.dyn_ema > 0:
        ns.direction = 'long'
        pickle.dump(ns.direction, open('direction', 'wb'))
    else:
        ns.direction = 'short'
        pickle.dump(ns.direction, open('direction', 'wb'))
    now = datetime.utcnow()
    now = now.replace(microsecond=0)
    # if ns.old_direction and ns.old_direction != ns.direction:  # reset the stop-outs on reversal.
    #     ns.reversal_flag = False
    #     pickle.dump(ns.reversal_flag, open('reversal_flag', 'wb'))
    #     ns.entry_price = ns.glob_future_price
    #     pickle.dump(ns.entry_price, open('entry_price', 'wb'))
    #     ns.last_trade_time = now
    #     pickle.dump(ns.last_trade_time, open('last_trade_time', 'wb'))

    percent_of_maximum = .15 # tested with max drawdown of strat.
    max_contracts = (((ns.glob_btc_balance - ns.contract_balance) * 20 * ns.entry_price)  #todo possibly use ns.glob_future_price
                     + (ns.contract_balance * 20 * ns.entry_price)) / 100.0
    max_contracts = math.floor(max_contracts * percent_of_maximum)
    if max_contracts == 0: max_contracts = 1


    # if ns.entry_price == 0:
    #     if ns.direction == "long":
    #         ns.entry_price = (1 + maximum_loss) * ns.glob_future_price
    #         pickle.dump(ns.entry_price, open('entry_price', 'wb'))
    #     else:
    #         ns.entry_price = (1 - maximum_loss) * ns.glob_future_price
    #         pickle.dump(ns.entry_price, open('entry_price', 'wb'))


    # if ns.reversal_flag:  # determine the recovery %.
    # if ns.last_trade_time is None:
    #     ns.last_trade_time = now
    time_since_last_trade = now - ns.last_trade_time
    minutes_since_last_trade = int(time_since_last_trade.days * 1440 + math.ceil(time_since_last_trade.seconds / 60.))
    if minutes_since_last_trade == 0: minutes_since_last_trade = 1
    max_since_last_trade = ns.entry_price
    min_since_last_trade = ns.entry_price
    if ns.direction == 'long':
        draw_down = (ns.glob_future_price - min_since_last_trade) / min_since_last_trade
                        # 1100                1000                    1000
    elif ns.direction == 'short':
        draw_down = (max_since_last_trade - ns.glob_future_price) / ns.glob_future_price
                              # 1100                      1000           1000
    # if not ns.reversal_flag:
    #     time_since_last_trade = now - ns.last_trade_time
    #     minutes_since_last_trade = int(
    #         time_since_last_trade.days * 1440 + math.ceil(time_since_last_trade.seconds / 60.))
    #     if minutes_since_last_trade == 0: minutes_since_last_trade = 1
    #     try:
    #         max_since_last_trade = max(ns.data[-minutes_since_last_trade:])
    #         min_since_last_trade = min(ns.data[-minutes_since_last_trade:])
    #     except TypeError as e:
    #         print(time_since_last_trade, minutes_since_last_trade, ns.last_trade_time)
    #         raise e
    #     if ns.direction == 'long':  # determine the stoploss %.
    #         draw_down = (ns.glob_future_price - max_since_last_trade) / max_since_last_trade
    #                        # 1000                1100                    1100
    #     elif ns.direction == 'short':
    #         draw_down = (min_since_last_trade - ns.glob_future_price) / ns.glob_future_price
                           # 1000                      1100           1100
    if now.second == 0:
        print(" entry price:", ns.entry_price, "Profit/Loss:", str(round(draw_down * 2500. * percent_of_maximum, 2)) + "%", ns.direction, "for", minutes_since_last_trade, "minutes")

    # determine if stoploss is below stoploss %
    # if not ns.reversal_flag and draw_down < maximum_loss:
    #     ns.reversal_flag = True
    #     pickle.dump(ns.reversal_flag, open('reversal_flag', 'wb'))
    #     ns.entry_price = ns.glob_future_price
    #     pickle.dump(ns.entry_price, open('entry_price', 'wb'))
    #     ns.last_trade_time = now
    #     pickle.dump(ns.last_trade_time, open('last_trade_time', 'wb'))
    #     emergency_mode = True
    #     print("HIT STOPLOSS EXITING ALL POSITIONS")
    #     desired_longs = 0
    #     desired_shorts = 0
    #
    # if ns.reversal_flag and draw_down < reentry_percent and (ns.glob_current_longs != 0 or ns.glob_current_shorts != 0):
    #     emergency_mode = True  # continue the sell-off after the first tick
    #     desired_longs = 0
    #     desired_shorts = 0
    # if ns.reversal_flag and draw_down < reentry_percent and ns.glob_current_longs == 0 and ns.glob_current_shorts == 0:
    #     desired_longs = 0
    #     desired_shorts = 0
    #     emergency_mode = False
    # if ns.reversal_flag:  # general stopped-out run
    #
    #     if ns.direction == 'long' and draw_down > reentry_percent:  # check if we should re-enter long
    #         emergency_mode = False
    #         desired_shorts = 0
    #         desired_longs = max_contracts
    #         ns.entry_price = ns.glob_future_price
    #         pickle.dump(ns.entry_price, open('entry_price', 'wb'))
    #         ns.last_trade_time = now
    #         pickle.dump(ns.last_trade_time, open('last_trade_time', 'wb'))
    #         ns.reversal_flag = False
    #         pickle.dump(ns.reversal_flag, open('reversal_flag', 'wb'))
    #         print("re-entering long position")
    #     if ns.direction == 'short' and draw_down > reentry_percent:  # check if we should re-enter short
    #         emergency_mode = False
    #         desired_shorts = max_contracts
    #         desired_longs = 0
    #         ns.entry_price = ns.glob_future_price
    #         pickle.dump(ns.entry_price, open('entry_price', 'wb'))
    #         ns.last_trade_time = now
    #         pickle.dump(ns.last_trade_time, open('last_trade_time', 'wb'))
    #         ns.reversal_flag = False
    #         pickle.dump(ns.reversal_flag, open('reversal_flag', 'wb'))
    #         print("re-entering short position")

    if ns.direction == 'short': # normal enter new long
        desired_shorts = max_contracts
        desired_longs = 0
        emergency_mode = False
        if ns.glob_current_shorts == 0 and ns.glob_btc_balance != 0 and ns.glob_current_shorts + ns.glob_current_longs != 0:
            ns.entry_price = ns.glob_future_price
            pickle.dump(ns.entry_price, open('entry_price', 'wb'))
            ns.last_trade_time = now
            pickle.dump(ns.last_trade_time, open('last_trade_time', 'wb'))
            notification("short", str(ns.glob_future_price) + "with" + str(max_contracts) + "contracts")
    elif ns.direction == 'long':  # normal enter new short
        desired_longs = max_contracts
        desired_shorts = 0
        emergency_mode = False
        if ns.glob_current_longs == 0 and ns.glob_btc_balance != 0 and ns.glob_current_shorts + ns.glob_current_longs != 0:
            ns.entry_price = ns.glob_future_price
            pickle.dump(ns.entry_price, open('entry_price', 'wb'))
            ns.last_trade_time = now
            pickle.dump(ns.last_trade_time, open('last_trade_time', 'wb'))
            notification("long", str(ns.glob_future_price) + "with" + str(max_contracts) + "contracts")
    if ns.reversal_flag :
        emergency_mode = False  # todo someday code the bit where it trades with prejudice if market goes real bad
    # else:
    #     desired_longs = 0
    #     desired_shorts = 0
    #     print("error in desired_position")

    # print("starting deltas")
    longs_delta = desired_longs - ns.glob_current_longs  # positive = get more longs
    shorts_delta = desired_shorts - ns.glob_current_shorts

    ns.old_direction = ns.direction
    return longs_delta, shorts_delta, emergency_mode


def trade(longs_delta, shorts_delta, emergency_mode, ns):
    global glob_futws
    futapikey = 'f042eed5-faa0-4f14-98d8-045d2c5acc4d'
    futsecretkey = '3D0432CD2D798DFA8D91CEEBC4C17F88'
    symbol = "btc_usd"
    contractType = "quarter"
    # place order at best limit position
    #print(ns.glob_orderbook)  #
    #print("trades:", ns.glob_trades)
    closest_ask = ns.glob_orderbook['asks'][19]  # gives list example [609.33, 30] ob going up
    closest_bid = ns.glob_orderbook['bids'][0]  # ex [608.89, 24] ob going down
    for item, line in enumerate(closest_ask):
        closest_ask[item] = float(line)
    for item, line in enumerate(closest_bid):
        closest_bid[item] = float(line)
    # 1 = place long, 2= place short, 3= liquidate long, 4 = liquidate short

    #cur_contracts = ns.book_contracts  # todo break book_contracts into per-trade-type
    possibilities = [
        # amount0, string1, trade type to generate2, opposite type3, existing orders4
        [max(-longs_delta, 0), 'close long', 3, 1, ns.book_contracts_close_long],
        [max(-shorts_delta, 0), 'close short', 4, 2, ns.book_contracts_close_short],
        [max(longs_delta, 0), 'open long', 1, 3, ns.book_contracts_open_long],
        [max(shorts_delta, 0), 'open short', 2, 4, ns.book_contracts_open_short],

    ]
    trades = copy.deepcopy(ns.glob_trades)
    # for name, listing in trades.items():
    #
    #     for i in [1, 2, 3, 4]:
    #         if int(listing['type']) == i:
    #             possibilities[i-1][4] += int(listing['amount'])
        # print(name, listing)
    if trades != {} or longs_delta != 0 or shorts_delta != 0:
        pass
        #print("trade called with ", longs_delta, shorts_delta, emergency_mode, "currentstuff =", ns.glob_current_longs, ns.glob_current_shorts)
        #print("                             ", possibilities)
    pennysnipe_bid = str(float(closest_bid[0]) + .01)
    pennysnipe_ask = str(float(closest_ask[0]) - .01)
    if ns.error_count > 10:  # if something has gone wrong, get a new understanding of the orderbook
        ns.error_count = -4
        while glob_futws.sock is None or not glob_futws.sock.connected:
            time.sleep(1)
        #glob_futws.send(future_get_orders(ns, futapikey, futsecretkey, symbol, contractType))
        glob_futws.send(future_trade(ns, futapikey, futsecretkey, symbol, contractType, price=str(ns.glob_future_price * .975),
                                     amount='1', tradeType=str(1), matchPrice='0', leverRate='20'))
        glob_futws.send(future_trade(ns, futapikey, futsecretkey, symbol, contractType, price=str(ns.glob_future_price * .975),
                                     amount='1', tradeType=str(4), matchPrice='0', leverRate='20'))
        glob_futws.send(future_trade(ns, futapikey, futsecretkey, symbol, contractType, price=str(ns.glob_future_price * 1.025),
                                     amount='1', tradeType=str(2), matchPrice='0', leverRate='20'))
        glob_futws.send(future_trade(ns, futapikey, futsecretkey, symbol, contractType, price=str(ns.glob_future_price * 1.025),
                                     amount='1', tradeType=str(3), matchPrice='0', leverRate='20'))
        time.sleep(2)
    if longs_delta == 0 and shorts_delta == 0 and trades != {}:
        ns.ghost_trade_counter += 1
    else:
        ns.ghost_trade_counter = 0

    if ns.ghost_trade_counter > 9:
        ns.glob_trades.clear()  # clear out my understanding of my open trades because ghost orders
        #glob_futws.send(future_get_orders(ns, futapikey, futsecretkey, symbol, contractType))
        ns.ghost_trade_counter = 0
    while glob_futws.sock is None or not glob_futws.sock.connected:
        time.sleep(1)
    glob_futws.send(future_get_orders(ns, futapikey, futsecretkey, symbol, contractType))

    while glob_futws.sock is None or not glob_futws.sock.connected:
        time.sleep(1)
    time.sleep(.7)  # main sleep tick, check pauseflag before and after
    for ordertype in possibilities:  # ordertype[2] % 2 == 0
        while glob_futws.sock is None or not glob_futws.sock.connected:
            time.sleep(1)
        i_own_the_best = False
        trades = copy.deepcopy(ns.glob_trades)
        # for name, listing in trades.items():
        #     for i in [1, 2, 3, 4]:
        #         possibilities[i - 1][4] = 0
        #         if int(listing['type']) == i:
        #             possibilities[i - 1][4] += int(listing['amount'])
        for name, listing in trades.items():
            while glob_futws.sock is None or not glob_futws.sock.connected:
                time.sleep(1)
            if closest_ask[0] - closest_bid[0] < .05 and int(listing['type']) == ordertype[2]:
                # cancel open order prep market buy
                glob_futws.send(
                    future_cancel_order(ns, futapikey, futsecretkey, symbol, str(listing['order_id']),
                                        contractType))
                print(ordertype[1], str(listing['amount']), "canceled limit order to prep for market order")
        if ordertype[0] > 0 or ordertype[4] > 0 and glob_futws.sock.connected:

            while glob_futws.sock is None or not glob_futws.sock.connected:
                time.sleep(1)
            if closest_ask[0] - closest_bid[0] < .05 and not emergency_mode and not len(trades) >= 1:  # market buy that which is possible
                if ordertype[0] - ordertype[4] > 0:
                    while glob_futws.sock is None or not glob_futws.sock.connected:
                        time.sleep(1)

                    if ordertype[2] == 1 or ordertype[2] == 4:  # buy charging up
                        glob_futws.send(future_trade(ns, futapikey, futsecretkey, symbol, contractType,
                                                     amount=str(min(closest_ask[1], ordertype[0] - ordertype[4])),
                                                     tradeType=str(ordertype[2]), matchPrice='1', leverRate='20'))

                        print(ordertype[1], str(min(closest_ask[1], ordertype[0])), "bought going up with market order")
                        time.sleep(1)
                    else:  # buy charging down
                        glob_futws.send(future_trade(ns, futapikey, futsecretkey, symbol, contractType,
                                                     amount=str(min(closest_bid[1], ordertype[0] - ordertype[4])),
                                                     tradeType=str(ordertype[2]), matchPrice='1', leverRate='20'))

                        print(ordertype[1], str(min(closest_bid[1], ordertype[0])), "Bought going down with market order")
                        time.sleep(1)
                while glob_futws.sock is None or not glob_futws.sock.connected:
                    time.sleep(1)
            elif not emergency_mode and not i_own_the_best and ordertype[0] - ordertype[4] > 0 and not len(trades) >= 1:
                # place limit order at penny-sniping price
                while glob_futws.sock is None or not glob_futws.sock.connected:
                    time.sleep(1)
                glob_futws.send(future_trade(ns, futapikey, futsecretkey, symbol, contractType,
                                             price=pennysnipe_bid if (
                                             ordertype[2] == 1 or ordertype[2] == 4) else pennysnipe_ask,
                                             amount=str(ordertype[0] - ordertype[4]),
                                             tradeType=str(ordertype[2]), matchPrice='0', leverRate='20'))
                print(ordertype[1], str(ordertype[0] - ordertype[4]), "normal limit order placed at pennysnipe")
                #ns.book_contracts += ordertype[0] - ordertype[4]
                time.sleep(1)
            elif emergency_mode:
                while glob_futws.sock is None or not glob_futws.sock.connected:
                    time.sleep(1)
                glob_futws.send(future_trade(ns, futapikey, futsecretkey, symbol, contractType, amount=str(ordertype[0]),
                                             tradeType=str(ordertype[2]), matchPrice='1', leverRate='20'))
                print("market bought", str(ordertype[0] - ordertype[4]), "emergency", ordertype[1])
                time.sleep(1)
            for name, listing in trades.items():
                # print(listing)
                if int(listing['type']) == ordertype[3] and ordertype[0] > 0:  # cancel existing open ~opposite~ orders
                    glob_futws.send(
                        future_cancel_order(ns, futapikey, futsecretkey, symbol, str(listing['order_id']),
                                            contractType))
                    print("canceling opposite order from", str(ordertype[1]), str(listing['amount']), )
                if int(listing['type']) == ordertype[2]:  # only canceling limit orders
                    while glob_futws.sock is None or not glob_futws.sock.connected:
                        time.sleep(1)
                    # print(listing)
                    if float(listing['price']) < closest_bid[0] and (
                                    ordertype[2] == 1 or ordertype[2] == 4):  # cancel buried open orders
                        glob_futws.send(
                            future_cancel_order(ns, futapikey, futsecretkey, symbol, str(listing['order_id']),
                                                contractType))
                        print(ordertype[1], str(listing['amount']), "canceled because lower than current bid")
                        # time.sleep(.3)
                    elif float(listing['price']) > closest_ask[0] and (
                                    ordertype[2] == 2 or ordertype[2] == 3):  # inverse context for buried asks
                        glob_futws.send(
                            future_cancel_order(ns, futapikey, futsecretkey, symbol, str(listing['order_id']),
                                                contractType))
                        print(ordertype[1], str(listing['amount']), "canceled because higher than current ask")
                        # time.sleep(.3)
                    elif i_own_the_best and (
                                    float(listing['price']) == closest_ask[0] or float(listing['price']) ==
                                closest_bid[0]):
                        glob_futws.send(
                            future_cancel_order(ns, futapikey, futsecretkey, symbol, str(listing['order_id']),
                                                contractType))
                        print(ordertype[1], str(listing['amount']), "canceled because duplicate")
                        # time.sleep(.3)
                    elif len(trades) >=2:


                        glob_futws.send(future_cancel_order(ns, futapikey, futsecretkey, symbol,
                                                                str(listing['order_id']), contractType))
                        print("duplicate", ordertype[1], "canceled")
                        #trades.pop(int(listing['order_id']), None)
                        break
                    elif float(listing['price']) == closest_ask[0] or float(listing['price']) == closest_bid[0]:
                        if ordertype[0] == 0:
                            glob_futws.send(
                                future_cancel_order(ns, futapikey, futsecretkey, symbol, str(listing['order_id'])
                                                    , contractType))
                            print("orphan", ordertype[1], "canceled")
                            # time.sleep(.3)
                        else:
                            i_own_the_best = True
                    elif ordertype[0] == 0:
                        glob_futws.send(
                            future_cancel_order(ns, futapikey, futsecretkey, symbol, str(listing['order_id']),
                                                contractType))
                        print("orphan", ordertype[1], "canceled")
                            # print("everything is good, I own the best", ordertype[1], str(listing['amount']), "on the books")




#
# def dtw(minute):
#     lookback = 100
#     look_forward = 100
#     ema_period = 10
#
#     query = data[-lookback:]
#     back_data = data[:-lookback]
#     np.savetxt("Data" + str(minute) + ".txt", back_data, delimiter='\t')
#     np.savetxt("Query" + str(minute) + ".txt", query, delimiter='\t')
#     # print(run(["wc", "Data.txt"], stdout=PIPE))  # TODO temp debug measure
#     runresult = run(
#         ["nice", "-5", "./UCR_DTW.compiled", "Data" + str(minute) + ".txt", "Query" + str(minute) + ".txt", str(lookback), ".12"],
#         stderr=DEVNULL, stdout=PIPE,
#         universal_newlines=True)
#     os.remove("Query" + str(minute) + ".txt")
#     os.remove("Data" + str(minute) + ".txt")
#     resultlines = runresult.stdout
#     resultlines = resultlines.split('\n')
#
#     for result in resultlines:
#         split = result.split(":")
#         if split[0] == 'Location ':
#             position = int(split[1])
#     global ns
#     average = np.mean(data[position + lookback + 1:position + lookback + look_forward])
#     # print("tadata came back", tadata[0][0])
#     ns.dyn_prediction = (average - data[position]) / data[position]
#     ns.dyn_ema = (ns.dyn_prediction - ns.dyn_ema) * (2 / (ema_period + 1)) + ns.dyn_ema
#     tempdisplay = float(ns.dyn_prediction * 1000)
#     tempdisplay2 = float(ns.dyn_ema * 1000)
#     print("ema", ns.dyn_ema) # todo temporary
#     print("1000", "{0:.00f}%".format(tempdisplay), ns.glob_future_price, "at position", position, datetime.now())

def evolved_predictor(ns, previous_minute, data):
    # old [0.0, 214.0, 0.0, 20819.0, 0.38980044000377911, 12554.0]
    # [1299.0, 1358.0, 0.0039968103821166586, 10.507419367673087, 4917.252866397499]
    #truncated_data = np.array(ns.data[-29964:])  # numpy array view, data is now pre-truncated before passing in
    truncated_data = np.array(data)
    datashape = truncated_data.shape
    #print(datashape)
    truncated_data = np.reshape(truncated_data, datashape[0])
    #print(truncated_data.shape)
    ema1 = ta.TEMA(truncated_data, timeperiod=475)
    #ns.stddev = max(np.std(truncated_data[-12554:]) / truncated_data[-1], .0001)
    ema2 = ta.EMA(truncated_data, timeperiod=9988)
    # ns.ema2 = ema2
    previous_ema = ns.dyn_ema
    time_since_last_trade = datetime.utcnow() - ns.last_trade_time
    minutes_since_last_trade = int(time_since_last_trade.days * 1440 + math.ceil(time_since_last_trade.seconds / 60.))
    if minutes_since_last_trade == 0: minutes_since_last_trade = 1
    # if previous_ema == -1 and ema1 > ema2:  #todo change to the whole min since last trade bit
    #     ns.reversal_flag = False
    #     pickle.dump(ns.reversal_flag, open('reversal_flag', 'wb'))
    # if previous_ema == 1 and ema1 < ema2:
    #     ns.reversal_flag = False
    #     pickle.dump(ns.reversal_flag, open('reversal_flag', 'wb'))
    low_threshold = ema2 * (1 - (.01687719*(38.14-(min(minutes_since_last_trade, 9958)/(9958/(38.14-1))))))
    high_threshold = ema2 * (1 + (.01687719*(38.14-(min(minutes_since_last_trade, 9958)/(9958/(38.14-1))))))
    # if not ns.reversal_flag:
    if ns.direction == 'long' or ns.direction is None:
        if data[-2] > low_threshold:
            ns.dyn_ema = 1
        else:
            ns.dyn_ema = -1
            # ns.reversal_flag = True
            # pickle.dump(ns.reversal_flag, open('reversal_flag', 'wb'))

    if ns.direction == 'short':
        if data[-2] < high_threshold:
            ns.dyn_ema = -1
        else:
            ns.dyn_ema = 1
            # ns.reversal_flag = True
            # pickle.dump(ns.reversal_flag, open('reversal_flag', 'wb'))
    # else:  # if reversal flag is set
    #     if ema1 > ema2:
    #         if ns.glob_future_price > high_threshold:
    #             ns.dyn_ema = 1
    #             ns.reversal_flag = False
    #             pickle.dump(ns.reversal_flag, open('reversal_flag', 'wb'))
    #         else:
    #             ns.dyn_ema = -1  #
    #
    #     if ema1 < ema2:
    #         if ns.glob_future_price < low_threshold:
    #             ns.dyn_ema = -1
    #             ns.reversal_flag = False
    #             pickle.dump(ns.reversal_flag, open('reversal_flag', 'wb'))
    #         else:
    #             ns.dyn_ema = 1
    if previous_minute != datetime.now().minute:
        print("current price:", ns.glob_future_price, "ema1 ema2", round(ema1, 2), round(ema2, 2), datetime.now())#, "reverse:", ns.reversal_flag)
        if ns.dyn_ema > 0:
            print("  next trade when crossover", round(low_threshold, 2))
        else:
            print("  next trade when crossover", round(high_threshold, 2))


def dyn_mash_and_predict(ns):
    # responsible for calculating the current DFW prediction, the EMA of such, and manage
    # the minute-close data on disk pulling from DB.


    previous_minute = None

    while True:
        time.sleep(.1)
        now = datetime.utcnow()
        now = now.replace(microsecond=0)



        globalvar.l.acquire()
        try:
            globalvar.dbcur.execute("SELECT close FROM futures ORDER BY time ASC")
            data = np.array(globalvar.dbcur.fetchall())
            data = np.array(data)
        except apsw.BusyError:
            #globalvar.l.release()
            print("got busyerror")
        finally:
            globalvar.l.release()
        if data is None: continue
        #data[-1] = ns.glob_future_price
        p = Process(target=evolved_predictor, args=(ns, previous_minute, data[-29964:]))#, args=(now.minute, ), name=str(now.minute))
        p.daemon = True
        p.start()

        previous_minute = now.minute
        time.sleep(5)
        if glob_futws.sock.connected is None:
            time.sleep(5)
            continue  # todo possibly should be break

# def plt_mash_and_predict(ns):
#     import build_ta_data
#     (plt_rf, oldname) = get_plt_forest(plt_rf=0, oldname=0)
#     new_forest_flag = False
#     while True:
#         now = datetime.utcnow()
#         now = now.replace(microsecond=0)
#         oneweekago = now - timedelta(weeks=1)
#         tadata = build_ta_data.dataproc_worker(oneweekago, now, True, ns)
#         averaged_current_price = tadata[0]
#         tadata = np.array(tadata).reshape((1, -1))
#         # print("tadata came back", tadata[0][0])
#         t1 = plt_rf.predict(tadata)
#         t1 = t1[0]
#         t1 /= 30  # important scaling factor reduction TODO testing for running 1-10 only, 3 should be normal-ish?
#         # print (temp)
#         ns.plt_prediction = t1
#         now = datetime.utcnow()
#         now = now.replace(microsecond=0)
#         pointstemp = copy.deepcopy(ns.glob_plotpoints)
#         if ns.plot_flag:
#             pointstemp.append([int(datetime.timestamp(now + timedelta(minutes=10))),
#                                [((t1 + averaged_current_price) * (ns.glob_future_price/averaged_current_price)), 't1', now]])
#
#             pointstemp.append([int(datetime.timestamp(now)), [ns.glob_future_price, 'real', now]])
#         else:
#             pointstemp.append([int(datetime.timestamp(now + timedelta(minutes=10))),
#                                [((t1 + averaged_current_price) * (ns.glob_future_price/averaged_current_price)), 'pred', now]])
#             pointstemp.append([int(datetime.timestamp(now)), [ns.glob_future_price, 'real', now]])        # get rid of old timestamps while I'm in here
#
#         ns.glob_plotpoints = pointstemp
#         pointstemp = copy.deepcopy(ns.glob_plotpoints)  # do another workaround
#         # print(len(pointstemp))
#         # print(int(datetime.timestamp(now)), now - timedelta(minutes=5))
#         if not ns.plot_flag:
#             for index, value in enumerate(ns.glob_plotpoints):
#
#                 if value[0] < int(datetime.timestamp(now)) and value[1][1] == 'pred':
#                     #print(index, value, int(datetime.timestamp(now-timedelta(minutes=1))))
#                     pointstemp.remove(value)
#                 elif value[0] < int(datetime.timestamp(now - timedelta(minutes=10))) and value[1][1] == 'real':
#                     # print(index, value, int(datetime.timestamp(now-timedelta(minutes=1))))
#                     pointstemp.remove(value)
#                 elif value[1][2] < now - timedelta(minutes=6) and value[1][1] == 'pred':
#                     pointstemp.remove(value)
#
#         time.sleep(.1)
#         # print(pointstemp)
#         ns.glob_plotpoints = pointstemp
#         pointstemp = 0
#
#         if now.minute % 2 == 0 and new_forest_flag:
#             (plt_rf, oldname) = get_plt_forest(plt_rf, oldname=oldname)
#             new_forest_flag = False
#         if now.minute % 2 == 1:
#             new_forest_flag = True
#         print("10-100","{0:.0f}%".format(t1* 100000), datetime.now())
#         # import pdb; pdb.set_trace()  # memory profiling
#         import objgraph


def on_error(self, evt):
    self.namespace.glob_pause_flag = True

    print(evt)

    raise evt


def on_close(self):
    print('DISCONNECTED')


# def get_dyn_forest(dyn_rf, oldname):
#     forestname = []
#     directory = os.listdir()
#     forests = fnmatch.filter(directory, 'forest*dyn.pkl')
#     for file in forests:
#         file = file[:-6]
#         file = file[6:-1]
#         file = int(file)
#         forestname.append(file)
#     forestname = max(forestname)
#     if forestname > oldname:
#         print("New dynamic forest high-score found, using", str(forestname))
#         dyn_rf = joblib.load("forest" + str(forestname) + "dyn.pkl")
#         oldname = forestname
#     return (dyn_rf, oldname)
#
# def get_plt_forest(plt_rf, oldname):
#     forestname = []
#     directory = os.listdir()
#     forests = fnmatch.filter(directory, 'forest*plt.pkl')
#     for file in forests:
#         file = file[:-6]
#         file = file[6:-1]
#         file = int(file)
#         forestname.append(file)
#     forestname = max(forestname)
#     if forestname > oldname:
#         print("New plotted forest high-score found, using", str(forestname))
#         plt_rf = joblib.load("forest" + str(forestname) + "plt.pkl")
#         oldname = forestname
#     return (plt_rf, oldname)

def future_websocket(ns):
    futURL = "wss://real.okex.com:10440/websocket/okcoinapi"
    global glob_futws
    while True:
        try:
            glob_futws = websocket.WebSocketApp(futURL,
                                                          on_message=fut_on_message,
                                                          on_error=on_error,
                                                          on_close=on_close, namespace=ns)
            glob_futws.on_open = fut_on_open
            glob_futws.on_ping = on_ping
            glob_futws.run_forever(ping_interval=6, ping_timeout=3)

        except KeyboardInterrupt:
            break

        else:
            continue
m = Manager()
ns = m.Namespace()

ns.glob_plotpoints = []
ns.glob_future_price = 0.0
ns.dyn_prediction = 0.0
ns.glob_btc_balance = 0.0
ns.glob_current_longs = 0
ns.glob_current_shorts = 0
glob_futws = websocket.WebSocketApp(url="no")
ns.glob_orderbook = m.dict()
ns.glob_trades = m.dict()  # todo both of these changed from standard dicts to threading aware managed dicts
ns.glob_last_tick = datetime.utcnow()
ns.glob_last_ob = datetime.utcnow()
ns.fitted_eq = None
ns.error_count = 0
# ns.plot_flag = False  #
ns.ema2 = 0
ns.dyn_ema = 0
ns.plt_prediction = 0
ns.ghost_trade_counter = 0
ns.contract_balance = 0
ns.book_contracts_close_long = 0
ns.book_contracts_close_short = 0
ns.book_contracts_open_long = 0
ns.book_contracts_open_short = 0

if os.path.isfile('reversal_flag'):
    ns.reversal_flag = pickle.load( open('reversal_flag', 'rb'))
    print("loaded reversal condition", ns.reversal_flag)
else:
    ns.reversal_flag = False
if os.path.isfile('entry_price'):
    ns.entry_price = pickle.load( open('entry_price', 'rb'))
    print("loaded entry price of", ns.entry_price)
else:
    ns.entry_price = 0
if os.path.isfile('last_trade_time'):
    ns.last_trade_time = pickle.load( open('last_trade_time', 'rb'))
    print("loaded last trade time of", ns.last_trade_time )
else:
    ns.last_trade_time = datetime.utcnow() - timedelta(minutes=5000)
if os.path.isfile('direction'):
    ns.direction = pickle.load(open('direction', 'rb'))
    print("loaded direction of ", ns.direction)
else:
    ns.direction = 'long'
ns.old_direction = None
ns.stddev = .01
ns.quarter_exists = True

def notification(value1, value2):
    report = {}
    report["value1"] = str(value1)
    report["value2"] = str(value2)
    requests.post("https://maker.ifttt.com/trigger/bot/with/key/d-Wg4XiIHzrLKgnHVkmaTt", data=report)


def main(ns):

    
    
    get_history.main(ns)
    fillgaps = Process(target=fill_gap,args=(ns, ), name="fillgaps")
    fillgaps.daemon = True
    fillgaps.start()


    # keepupdated = Process(target=keepdbupdated,args=(ns,), name="keepupdated")
    # keepupdated.daemon = True
    # keepupdated.start()
    futurewebsocket = threading.Thread(target=future_websocket, args=(ns,), name="futurewebsocket")
    futurewebsocket.setDaemon(True)
    futurewebsocket.start()

    dynmashandpredict = Process(target=dyn_mash_and_predict, args=(ns,), name="dynmashandpredict")
    dynmashandpredict.daemon = False
    dynmashandpredict.start()
    # pltmashandpredict = Process(target=plt_mash_and_predict, args=(ns,), name="pltmashandpredict")
    # pltmashandpredict.daemon = True
    # pltmashandpredict.start()
    while ns.dyn_ema == 0.0:
        time.sleep(1)
    orderbook = copy.deepcopy(ns.glob_orderbook)
    while orderbook == {}:
        time.sleep(1)
        orderbook = copy.deepcopy(ns.glob_orderbook)
    # while ns.data is None:
    #     time.sleep(1)
    print("starting main loop in 10 seconds")
    time.sleep(10)

    # while ns.glob_plotpoints == []:
    #     time.sleep(1)
    try:
        while True:
            while glob_futws.sock is None or not glob_futws.sock.connected:
                time.sleep(1)

            longs_delta, shorts_delta, emergency_mode = desired_position(ns)
            while glob_futws.sock is None or not glob_futws.sock.connected:
                time.sleep(1)
            trade(longs_delta, shorts_delta, emergency_mode, ns)
            ns.book_contracts_close_long = 0
            ns.book_contracts_close_short = 0
            ns.book_contracts_open_long = 0
            ns.book_contracts_open_short = 0
            trades = copy.deepcopy(ns.glob_trades)
            if len(trades) > 0:
                for name, open_order in trades.items():
                    if open_order['type'] == 1:
                        ns.book_contracts_open_long += int(open_order['amount'] - int(open_order['deal_amount']))
                    elif open_order['type'] == 2:
                        ns.book_contracts_open_short += int(open_order['amount'] - int(open_order['deal_amount']))
                    elif open_order['type'] == 3:
                        ns.book_contracts_close_long += int(open_order['amount'] - int(open_order['deal_amount']))
                    elif open_order['type'] == 4:
                        ns.book_contracts_close_short += int(open_order['amount'] - int(open_order['deal_amount']))

                    #ns.book_contracts += int(contract['amount']) - int(contract['deal_amount'])
            # print(datetime.utcnow() - ns.glob_last_ob, datetime.utcnow() - ns.glob_last_tick, ns.glob_pause_flag, "going into desired")
            if datetime.utcnow() - ns.glob_last_ob > timedelta(seconds=30):
                print("orderbook timed out")
                glob_futws.close()

            if datetime.utcnow() - ns.glob_last_tick > timedelta(seconds=60):
                print("tick timed out")
                glob_futws.close()


    finally:
        errorstring = ""
        for item in sys.exc_info():
            errorstring += str(item)
        notification("PROGRAM CRASHED ERROR ERROR ", errorstring)
        print("exited")
        pickle.dump(ns.entry_price, open('entry_price', 'wb'))
        pickle.dump(ns.direction, open('direction', 'wb'))
        pickle.dump(ns.last_trade_time, open('last_trade_time', 'wb'))
        raise BaseException
        futurewebsocket.join()
        dynmashandpredict.join()

        # trade(future, "closeall", old_price_dir) # TODO: figure out better close behavior and implement it.
        # TODO addendum: crashing cleanly without altering trades seems good, run the whole thing with a relauncher
        # TODO before going live, double check all <> and if logic


if __name__ == '__main__':
    # plt_mash_and_predict(ns)
    main(ns)
