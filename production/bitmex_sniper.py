testing = False

# from bravado.client import SwaggerClient
# from bravado.requests_client import RequestsClient

# from BitMEXAPIKeyAuthenticator import APIKeyAuthenticator
# from bitmex_websocket import BitMEXWebsocket
import datetime
import os
import pickle
import threading
from subprocess import run, PIPE, DEVNULL
from time import sleep

import numpy as np
import pytz
import pywt
from dateutil import parser
import talib.stream as ta

import get_history_bitmex
import globalvar
from market_maker import bitmex
from market_maker.utils import log
import api_keys

# import matplotlib.pyplot as plt


logger = log.setup_custom_logger('root', log_level='INFO')
if not testing:
    HOST = "https://www.bitmex.com"
    SPEC_URI = HOST + "/api/explorer/swagger.json"

    # See full config options at http://bravado.readthedocs.io/en/latest/configuration.html
    config = {
        # Don't use models (Python classes) instead of dicts for #/definitions/{models}
        'use_models': True,
        # This library has some issues with nullable fields
        'validate_responses': False,
        # Returns response in 2-tuple of (body, response); if False, will only return body
        'also_return_response': False,
    }

    API_KEY = api_keys.bmx_api_key
    API_SECRET = api_keys.bmx_api_secret

    bm = bitmex.BitMEX(base_url='https://www.bitmex.com/api/v1/', symbol='XBTUSD', login=None,
                       password=None, otpToken=None, apiKey=API_KEY,
                       apiSecret=API_SECRET, orderIDPrefix='jose')
else:  # testing stuff
    HOST = "https://testnet.bitmex.com"
    # SPEC_URI = HOST + "/api/explorer/swagger.json"

    # See full config options at http://bravado.readthedocs.io/en/latest/configuration.html
    config = {
        # Don't use models (Python classes) instead of dicts for #/definitions/{models}
        'use_models': True,
        # This library has some issues with nullable fields
        'validate_responses': False,
        # Returns response in 2-tuple of (body, response); if False, will only return body
        'also_return_response': False,
    }

    API_KEY = api_keys.testnet_api_key  # testnet
    API_SECRET = api_keys.testnet_api_secret

    bm = bitmex.BitMEX(base_url='https://testnet.bitmex.com/api/v1/', symbol='XBTUSD', login=None,
                       password=None, otpToken=None, apiKey=API_KEY,
                       apiSecret=API_SECRET, orderIDPrefix='jose')
# time_till_next_settlement = datetime.timedelta(hours=1)

symbol = "XBTUSD"
# ('returns, correct percentage, W, l, target, decomp, wavelet pad, wavelet, initial, knn, ema_time', '230.93', 0.5475, 0.13518536075870957, 62, 18, 5, 'symmetric', 'db3', 1, 14, 38)

# constants trained for KNN-DTW-DWT #
W = .13518536075870957  # warp factor
L = 62  # sequence length
future_target = 18  # targets average of n+1 to n+future_target
decomp_level = 5  # perform DWT this many levels
wavelet = 'db3'  # with this wavelet
wavelet_pad_mode = "symmetric"  # and this type of padding on the end of the sequences
wavelet_initial = 1  # drop all but 1 levels of the decomp factors to smooth
num_neighbors = 4  # grab nearest 4 neighbors
ema_timeperiod = 38  # smooth the chaotic results to get best trade indicator


def desired_contracts_func(desired_direction='long', postfunding=False, leverage_desired=45):
    logger.info("desired contracts called with " + desired_direction)

    XBTUSD = bm.instrument(symbol="XBTUSD")
    user_margin = bm.funds()
    # current_contracts = bm.position(symbol)['currentQty']
    if leverage_desired > 20:
        xbt_amount = min(user_margin['marginBalance'] / 100000000, 2)
    else:
        xbt_amount = user_margin['marginBalance'] / 100000000

    fair_price = min(  # XBTUSD['fairPrice'],
        XBTUSD['lastPrice'], XBTUSD['impactBidPrice'], XBTUSD['impactAskPrice'])
    # mark_price = XBTUSD['markPrice']
    # leverage_desired = 45  # dunno lol
    if postfunding:
        leverage_desired = 80
        if user_margin['availableMargin'] * 1.3 > user_margin['marginBalance']:  # if no pre-funding trade 1:1 ish
            # if pre-funding trade then available is much smaller than margin
            leverage_desired = 40
        xbt_amount = min(user_margin['availableMargin'] / 100000000, 2)
    fee_adjustment = 1 - (.00075 * leverage_desired)

    desired_contracts_outer = int(xbt_amount * fair_price * leverage_desired * fee_adjustment)

    if desired_direction == "short" and not postfunding: desired_contracts_outer *= -1
    if desired_direction == "long" and postfunding: desired_contracts_outer *= -1
    # if postfunding: # not necessary, using available margin already includes existing orders
    #     desired_contracts_outer += current_contracts
    #     if desired_direction == 'long' and desired_contracts_outer > 0:
    #         desired_contracts_outer = 0
    #     if desired_direction == 'short' and desired_contracts_outer < 0:
    #         desired_contracts_outer = 0
    return_thing = int(desired_contracts_outer)
    if return_thing is None:
        raise ValueError("returned None from desired_contracts_func")
    logger.info("desired contracts: " + str(return_thing))

    return return_thing


def determine_minutes_before_funding(funding_rate):
    # output from plot_bitmex_funding:
    # for 0.003 use T-3 for 5 % returns play the post-funding: True
    # for -0.003 use T-6 for 7 % returns play the post-funding: True
    # for 0.002 use T-3 for 1 % returns play the post-funding: False
    # for -0.002 use T-1 for 4 % returns play the post-funding: True
    # for 0.0015 use T-2 for 3 % returns play the post-funding: True
    # for -0.0015 use T-7 for 13 % returns play the post-funding: False
    # for 0.001 use T-9 for 5 % returns play the post-funding: False
    # for -0.001 use T-4 for 3 % returns play the post-funding: False
    # for 0.0005 use T-2 for 1 % returns play the post-funding: False
    # for -0.0005 use T-8 for 7 % returns play the post-funding: False

    minutes = 0
    post_fund_trade = False

    if abs(funding_rate) > .0030:
        if funding_rate > 0:
            minutes = 3
            post_fund_trade = True
        else:
            minutes = 6
            post_fund_trade = True
    elif abs(funding_rate) > .0020:
        if funding_rate > 0:
            minutes = 3
            post_fund_trade = False
        else:
            minutes = 1
            post_fund_trade = True
    elif abs(funding_rate) > .0015:
        if funding_rate > 0:
            minutes = 2
            post_fund_trade = True
        else:
            minutes = 7
            post_fund_trade = False
    elif abs(funding_rate) > .0010:
        if funding_rate > 0:
            minutes = 9
            post_fund_trade = False
        else:
            minutes = 4
            post_fund_trade = False
    elif abs(funding_rate) > .0005:
        if funding_rate > 0:
            minutes = 2
            post_fund_trade = False
        else:
            minutes = 8
            post_fund_trade = False


    return minutes, post_fund_trade


def DWT_smooth(initial):  # todo add soft smoothing boolean
    # http://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html?highlight=soft
    global decomp_level
    if wavelet is None:
        return initial
    try:
        coeffs = pywt.wavedec(initial, wavelet=wavelet, mode=wavelet_pad_mode, level=decomp_level)
    except ValueError as e:

        decomp_level = int(e.args[0][-2])
        coeffs = pywt.wavedec(initial, wavelet=wavelet, mode=wavelet_pad_mode, level=decomp_level)

    denoised = []
    for index, item in enumerate(coeffs):
        if index < wavelet_initial:  # use this many initial levels of coefficients
            denoised.append(item)
        else:
            denoised.append(None)

    smoothed = pywt.waverec(coeffs=denoised, wavelet=wavelet, mode=wavelet_pad_mode)
    return smoothed


def run_DTW(data):
    smoothed = DWT_smooth(data)
    # print(len(smoothed) - len(to_smooth))
    difference = len(smoothed) - len(data)
    least_distance = float('inf')
    best_offset = 0
    for iteration in range(difference * 2):
        if iteration == 0:
            smooth_to_compare = smoothed[-L:]
        else:
            smooth_to_compare = smoothed[-(L + iteration):-iteration]
        comparison = data[-L:]
        distance = np.linalg.norm(smooth_to_compare - comparison)
        if distance < least_distance:
            best_offset = iteration
            least_distance = distance
    # while len(smoothed) > len(data):
    #     smoothed = np.delete(smoothed, -1)
    # comparison = data[-L:]
    # query = smoothed[-L:]
    # plt.plot(comparison, "r-") # red is base
    # plt.plot(query, "k-")  # black is smooth
    # print("best offset", best_offset)

    # comparison = data[:200]
    # query = smoothed[:200]
    # plt.figure(3)
    # plt.plot(comparison, "r-")  # red is base
    # plt.plot(query, "k-")  # black is smooth
    # plt.show()

    # query = smoothed[-L:]
    if best_offset == 0:
        query = smoothed[-L:]
    else:
        query = smoothed[-(L + best_offset):-best_offset]
    # plt.figure(2)
    # comparison = data[-L:]

    # plt.plot(comparison, "r-")  # red is base
    # plt.plot(query, "k-")  # black is smooth
    # print("best offset", best_offset)
    # plt.show()
    data_to_search = data[:-(L + best_offset)]

    np.savetxt("Query.txt", query, delimiter='\t')
    np.savetxt("Data.txt", data_to_search, delimiter='\t')

    runresult = run(["nice", "-15", "./UCR_DTW_KNN.so", "Data.txt", "Query.txt", str(L), str(W), str(num_neighbors)],
                    stderr=DEVNULL, stdout=PIPE,
                    universal_newlines=True)
    os.remove("Query.txt")
    os.remove("Data.txt")
    # print(runresult)
    # position, distance = _ucrdtw.ucrdtw(smoothX, smoothmainX[len(mainX) - len(testX) + index - L: len(mainX) - len(testX) + index], L, W)
    resultlines = runresult.stdout
    resultlines = resultlines.split('\n')
    distancelist = []
    distancelist = resultlines[3:3 + num_neighbors]
    # for result in resultlines:
    #     split = result.split(":")
    #     if split[0] == 'Location ':
    #         position = int(split[1])
    #     if split[0] == 'Distance ':
    #         distance = float(split[1])
    for index2, item in enumerate(distancelist):
        distancelist[index2] = int(item)
    # if distance is None or (distance == 0 and position == 0):
    #     distance = 1337
    #     return (str(resultlines), distance, index)
    # if position == 0 or position is None:
    #     position = 1337

    return distancelist, data_to_search


recalculate_flag = False


def main():
    # for item in dir(bm):
    #     print(item)
    # ema1 = []
    # ema2 = []
    previous_calc_time = None
    global funding_has_control
    # recalculate_flag = False
    old_desired_direction = False
    global desired_contracts, use_market, previous_trade
    # desired_contracts = 0
    trade_thread = threading.Thread(target=trade)
    trade_thread.daemon = True
    trade_thread.start()
    predict_thread = threading.Thread(target=longterm)
    predict_thread.daemon = True
    predict_thread.start()
    # prediction_ema = 0
    sleep(10)
    while True:
        premature_close = False
        while bm.ws.exited or not bm.ws.wst.is_alive():
            bm.reconnect()
            sleep(1)

        XBTUSD = bm.instrument(symbol="XBTUSD")
        next_settlement_time = parser.parse(XBTUSD['fundingTimestamp'])
        next_settlement_funding = XBTUSD['fundingRate']
        now = datetime.datetime.now(tz=pytz.UTC)
        time_till_next_settlement = next_settlement_time - now
        user_margin = bm.funds()
        pre_trade_funds = user_margin['marginBalance']
        # choose to enter into the snipe, every 8 hours yes/no. Function is funding * 1000 then squared, ranges from 4 to 22min
        minutes_before_funding, post_fund_trade = determine_minutes_before_funding(next_settlement_funding)
        if not predict_thread.is_alive():
            raise ValueError("predict thread crashed probably nans")
        if not trade_thread.is_alive():
            raise ValueError("trade thread crashed probably compared to None")
        if time_till_next_settlement < datetime.timedelta(minutes=minutes_before_funding) or \
                time_till_next_settlement < datetime.timedelta(seconds=10):  # or testing:
            if minutes_before_funding != 0:
                pre_fund_trade = True
            else:
                pre_fund_trade = False
            if pre_fund_trade:
                funding_has_control = True
                if next_settlement_funding < 0: desired_direction = "long"
                if next_settlement_funding > 0: desired_direction = "short"
                desired_contracts_outer = desired_contracts_func(desired_direction)
                visible_rate = (abs(next_settlement_funding) - .0005) * 4500
                logger.info("going " + desired_direction + " with " + str(desired_contracts_outer) + " contracts to catch "
                            + str(next_settlement_funding * 100) + "% funding " + str(visible_rate) +
                            "% leveraged funding")
                now = datetime.datetime.now(tz=pytz.UTC)
                while bm.ws.exited or not bm.ws.wst.is_alive():
                    bm.reconnect()
                    sleep(1)
                time_till_next_settlement = next_settlement_time - now
                old_price = bm.instrument(symbol)['lastPrice']  # this v part delays until more favorable order conditions inc stalled
                while time_till_next_settlement > datetime.timedelta(minutes=2):
                    while bm.ws.exited or not bm.ws.wst.is_alive():
                        bm.reconnect()
                        sleep(1)
                    sleep(10)
                    current_price = bm.instrument(symbol)['lastPrice']
                    if (old_price <= current_price) == (desired_direction == 'long'):  # break (and start placing orders)
                        # if long and new > old, OR short and new < old
                        break
                    logger.info("sleeping until favorable market conditions")
                    now = datetime.datetime.now(tz=pytz.UTC)
                    time_till_next_settlement = next_settlement_time - now
                    old_price = current_price

                if testing:
                    with lock:
                        desired_contracts = desired_contracts_outer
                    sleep(5)
                else:
                    while bm.ws.exited or not bm.ws.wst.is_alive():
                        bm.reconnect()
                        sleep(1)
                    with lock:
                        desired_contracts = desired_contracts_outer
            # at this point I should have a successfully executed buy/sell
            now = datetime.datetime.now(tz=pytz.UTC)
            while now.minute != 0:
                sleep(.0001)
                now = datetime.datetime.now(tz=pytz.UTC)
                while bm.ws.exited or not bm.ws.wst.is_alive():
                    bm.reconnect()
                    sleep(1)
                position = bm.position(symbol)
                XBTUSD = bm.instrument(symbol)
                if pre_fund_trade:
                    if position['avgEntryPrice'] is None or position['avgEntryPrice'] == 0.0:
                        position['avgEntryPrice'] = XBTUSD['lastPrice']
                    if position['bankruptPrice'] is None:
                        position['bankruptPrice'] = float('inf')
                    # should margin call at 66% loss, or 2 parts deviation from entry 1 part deviation from mc price
                    if abs(XBTUSD['lastPrice'] - position['avgEntryPrice']) * .5 > abs(
                                    XBTUSD['lastPrice'] - position['bankruptPrice']) or testing:
                        logger.info("MARGIN CALLING last price " + str(XBTUSD['lastPrice']) + " entry price " +
                                    str(position['avgEntryPrice']) + " bankrupt price " + str(position['bankruptPrice']))
                        premature_close = True
                        break
                    if testing:
                        now = datetime.datetime(2017, 10, 10, hour=5, minute=0)
            # close position and also go balls deep the other way if post_fund_trade
            if post_fund_trade and not premature_close:
                funding_has_control = True
                while bm.ws.exited or not bm.ws.wst.is_alive():
                    bm.reconnect()
                    sleep(1)
                desired_contracts_outer = desired_contracts_func(desired_direction, postfunding=True)
                with lock:
                    desired_contracts = desired_contracts_outer
                    use_market = True
                old_price = bm.instrument(symbol)['lastPrice']
                sleep(30)
                current_price = bm.instrument(symbol)['lastPrice']

                while (old_price <= current_price) != (
                            desired_direction == 'long'):  # waits until short term trend is over/stalled
                    # if short and new > old, OR long and new < old  # opposite of above
                    while bm.ws.exited or not bm.ws.wst.is_alive():
                        bm.reconnect()
                        sleep(1)
                    old_price = current_price
                    logger.info("sleeping until favorable market conditions")
                    sleep(30)
                    current_price = bm.instrument(symbol)['lastPrice']
                    now = datetime.datetime.now(tz=pytz.UTC)
                desired_contracts_outer = previous_trade
                # now close the reverse position gracefully
                if testing:
                    sleep(5)
                    with lock:
                        desired_contracts = desired_contracts_outer
                else:
                    while bm.ws.exited or not bm.ws.wst.is_alive():
                        bm.reconnect()
                        sleep(1)
                    with lock:
                        desired_contracts = desired_contracts_outer
            else:
                desired_contracts_outer = previous_trade
                while bm.ws.exited or not bm.ws.wst.is_alive():
                    bm.reconnect()
                    sleep(1)
                with lock:
                    desired_contracts = desired_contracts_outer
                    use_market = True
                sleep(5)
            sleep(30)
            post_trade_funds = bm.funds()['marginBalance']
            percent_profit = ((post_trade_funds - pre_trade_funds) / pre_trade_funds) * 100
            logger.info("Profit for this execution was " + str(round(percent_profit, 2)) + "% " +
                        str(round((post_trade_funds - pre_trade_funds) / 100000000, 4)) + "XBT " +
                        str(round((post_trade_funds - pre_trade_funds) / 100000000 * XBTUSD['lastPrice'], 2)) + "$")
            if premature_close: break
            funding_has_control = False
            sleep(1)
        if premature_close:
            break
        sleep(1)
        funding_has_control = False


lock = threading.Lock()
if os.path.isfile('previous_trade'):
    previous_trade = pickle.load(open('previous_trade', 'rb'))
else:
    previous_trade = 0
desired_contracts = previous_trade
use_market = False
exec_inst = None
funding_has_control = False


def longterm():
    # part where I manage the non-funding long/shorts using 1054, 373 dual ema
    global desired_contracts, recalculate_flag, previous_trade
    # prediction_ema = 0
    old_desired_direction = 'none'
    previous_calc_time = datetime.datetime.now(tz=pytz.UTC) - datetime.timedelta(minutes=200)
    ema1 = []
    ema2 = []
    recalculate_flag = False
    old_desired_direction = None

    if os.path.isfile('previous_trade'):
        previous_trade = pickle.load(open('previous_trade', 'rb'))
        if previous_trade > 0:
            old_desired_direction = 'long'
        else:
            old_desired_direction = 'short'
    else:
        previous_trade = None
        recalculate_flag = True
    # time_since_prev_calc = datetime.datetime.now(tz=pytz.UTC) - previous_calc_time
    # num_to_calculate = int(time_since_prev_calc.seconds / 60)
    # restart_flag = False
    sleep(10)
    while True:

        now = datetime.datetime.now(tz=pytz.UTC)
        if previous_calc_time and now.minute == previous_calc_time.minute and not recalculate_flag:
            sleep(1)
            continue

        while bm.ws.exited or not bm.ws.wst.is_alive():
            bm.reconnect()
            sleep(1)
        if not testing:
            with lock:
                while not get_history_bitmex.main(bm):
                    pass
        time_since_prev_calc = datetime.datetime.now(tz=pytz.UTC) - previous_calc_time
        # num_to_calculate = int(time_since_prev_calc.seconds / 60)
        previous_calc_time = datetime.datetime.now(tz=pytz.UTC)
        with lock:
            globalvar.dbcur.execute("SELECT * FROM bmxswap ORDER BY time DESC limit 1427")
            data = globalvar.dbcur.fetchall()
            data = np.array(data[::-1])
        # while num_to_calculate > 0:

        while len(ema1) < 373:
            offset = 373 - len(ema1)
            truncated_data = data[-offset - 1054: -offset]
            ema1.append(ta.EMA(truncated_data[:, 4], timeperiod=1054))
        while len(ema2) < 3:
            offset = 3 - len(ema2)
            truncated_data_ema1 = ema1[-offset - 373: -offset]
            ema2.append(ta.EMA(np.array(truncated_data_ema1), timeperiod=370))
        if old_desired_direction is None:
            if ema2[-1] > ema2[-2]:
                old_desired_direction = 'long'
            else:
                old_desired_direction = 'short'
        prediction_ema = ema2[-1] > ema2[-2]
        # nearest_neighbors, smoothed_data = run_DTW(data[:-num_to_calculate, 4])
        # # smoothed_data = DWT_smooth(data[:, 4])
        # average = []
        # for position in nearest_neighbors:
        #     average.append((np.mean(smoothed_data[position + L + 1:position + L + future_target])
        #                     - smoothed_data[position + L]) / smoothed_data[position + L])
        # average = np.mean(average)
        # prediction_ema = (average - prediction_ema) * (
        #     2 / (ema_timeperiod + 1)) + prediction_ema  # manually calculating ema
        # num_to_calculate -= 1
            # logger.info("prediction_ema:"+str(prediction_ema)+" num_calc "+str(num_to_calculate)) 
            # if num_to_calculate >= 2:
            #     if num_to_calculate % 10 == 0:
            #         logger.info("starting up " +
            #                     str(round(num_to_calculate*100/int(time_since_prev_calc.seconds / 60), 2)) + "%")
            #     recalculate_flag = True
            #     restart_flag = True
        # if restart_flag:
        #     restart_flag = False
        #     continue

        if prediction_ema > 0:  # 
            desired_direction = 'long'
            if desired_direction != old_desired_direction:
                logger.info("longterm going long")
                recalculate_flag = True
            logger.debug("continuing going long")
        else:
            desired_direction = 'short'
            if desired_direction != old_desired_direction:
                logger.info("longterm going short")
                recalculate_flag = True
            logger.debug("continuing going short")
        if np.nan in np.array(ema1):
            logger.critical("got NAN in prediction_ema, restarting")
            # trade(0)
            with lock:
                desired_contracts = 0
                sleep(10)
            while bm.position(symbol)['currentQty'] != 0:
                sleep(1)
                pass
            raise ValueError("got NAN in prediction_EMA")

        if recalculate_flag:
            desired_contracts_longterm = desired_contracts_func(desired_direction=desired_direction, leverage_desired=5)
            previous_trade = desired_contracts_longterm
            recalculate_flag = False
            pickle.dump(desired_contracts_longterm, open('previous_trade', 'wb'))
        else:
            desired_contracts_longterm = previous_trade
        # num_to_calculate += 1
        while bm.ws.exited or not bm.ws.wst.is_alive():
            bm.reconnect()
            sleep(1)
        # trade(desired_contracts_outer, longterm=True)
        if funding_has_control is False:
            with lock:
                desired_contracts = desired_contracts_longterm
        # finishing up
        del ema1[0]
        del ema2[0]
        old_desired_direction = desired_direction
        recalculate_flag = False


def trade():
    global desired_contracts, use_market
    sleep(10)
    while True:
        try:
            current_contracts = bm.position(symbol)['currentQty']
        except KeyError:
            current_contracts = 0
        if current_contracts != desired_contracts:
            logger.info("trade called with desired " + str(desired_contracts))
        sleep(.5)
        orderid = None
        while bm.ws.exited or not bm.ws.wst.is_alive():
            bm.reconnect()
            sleep(1)
        # if desired_contracts_func < 0: desired_direction = "short"
        # elif desired_contracts_func > 0: desired_direction = "long"
        # else:
        submitted_order_flag = False
        old_order_buy_quantity = 0
        old_order_sell_quantity = 0
        # current_contracts = bm.position(symbol)['currentQty']
        # if current_contracts < 0: desired_direction = "long"
        # else: desired_direction = "short"
        try:
            bm.position(symbol)
        except KeyError:
            bm.reconnect()
            sleep(1)
            continue
        # sleep(40) # todo temp debugging
        while bm.position(symbol)[
            'currentQty'] != desired_contracts:  # this loop executes until we're at desired amount
            # place order for shorts/longs
            old_desired_contracts = desired_contracts
            bidprice, askprice = bm.best_bid_ask()
            bidprice, askprice = round(bidprice, 1), round(askprice, 1)
            orderids = bm.open_orders()
            if orderids != []:
                orderid = orderids[0]
            # this block detects existing open opposite orders and cancels them
            if orderid is not None:
                if desired_contracts > bm.position(symbol)['currentQty']:
                    if orderid['side'] == "Buy":
                        bm.cancel(orderid['orderID'])  # cancel sell orders if I want to add contracts
                else:
                    if orderid['side'] == "Sell":
                        bm.cancel(orderid['orderID'])  # cancel buy orders if I want to subtract contracts
                if desired_contracts != old_desired_contracts:  # check if desired contracts has changed
                    bm.cancel(orderid['orderID'])
                    sleep(.5)
            try:
                if submitted_order_flag is False or bm.position(symbol)['openOrderBuyQty'] != old_order_buy_quantity \
                        or bm.position(symbol)['openOrderSellQty'] != old_order_sell_quantity:
                    open_order_buy_quantity = bm.position(symbol)['openOrderBuyQty']
                    open_order_sell_quantity = bm.position(symbol)['openOrderSellQty']
                    old_order_buy_quantity = bm.position(symbol)['openOrderBuyQty']
                    old_order_sell_quantity = bm.position(symbol)['openOrderSellQty']
            except KeyError:
                bm.reconnect()
                sleep(1)
                continue

            if orderid is None or orderid['ordStatus'] == "Canceled":
                if desired_contracts > bm.position(symbol)['currentQty']:  # if going long
                    quantity = desired_contracts - bm.position(symbol)['currentQty'] - open_order_buy_quantity
                    if quantity != 0:
                        if use_market:
                            logger.info("long market order placed for " + str(quantity))
                            with lock:
                                use_market = False
                            orderid = bm.place_order(quantity=quantity,
                                                     exec_inst=exec_inst)  # market ordering directly after funding
                            open_order_buy_quantity += quantity
                            sleep(15)
                            # force exec_inst to None to override the default participatenotinit during market orders

                            continue
                        else:
                            orderid = bm.place_order(quantity=quantity, price=bidprice)
                        logger.info("long limit order placed")
                        open_order_buy_quantity += quantity
                        submitted_order_flag = True
                elif desired_contracts < bm.position(symbol)['currentQty']:  # going short
                    quantity = desired_contracts - bm.position(symbol)['currentQty'] + open_order_sell_quantity
                    if quantity != 0:
                        if use_market:
                            logger.info("short market order placed for " + str(quantity))
                            with lock:
                                use_market = False
                            orderid = bm.place_order(quantity=quantity,
                                                     exec_inst=exec_inst)  # market ordering directly after funding
                            open_order_sell_quantity -= quantity
                            sleep(15)

                            continue
                        else:
                            orderid = bm.place_order(quantity=quantity, price=askprice)
                        logger.info("short limit order placed")
                        open_order_sell_quantity -= quantity
                        submitted_order_flag = True
                        # bitMEXAuthenticated.Order.Order_new(symbol='XBTUSD', orderQty=3, price=1000).result()
            sleep(.5)
            if orderid is None or not orderid['leavesQty'] > 0:
                orderid = None
            if bm.open_orders() == []:
                sleep(2)
                if bm.open_orders() == []:
                    submitted_order_flag = False
            # had an issue of double-ordering for full quantity. Added temp position augmentation and lengthened this delay.

            while bm.open_orders() != []:  # while I have open orders
                while bm.ws.exited or not bm.ws.wst.is_alive():
                    bm.reconnect()
                    sleep(1)
                bidprice, askprice = bm.best_bid_ask()
                bidprice, askprice = round(bidprice, 1), round(askprice, 1)

                if bm.open_orders()[0]['ordStatus'] == 'Canceled' or bm.open_orders()[0]['ordStatus'] == 'Filled':
                    orderid = None
                    continue
                orderids = bm.open_orders()
                for orderid in orderids:
                    if len(orderids) > 1:
                        logger.error(str(orderids))
                        bm.cancel(orderid['orderID'])
                        continue
                    now = datetime.datetime.now(tz=pytz.UTC)
                    if use_market:
                        bm.cancel(orderid['orderID'])
                        continue

                    if desired_contracts != old_desired_contracts:  # check if desired contracts has changed
                        bm.cancel(orderid['orderID'])
                        continue
                    if (orderid['price'] > askprice and orderid['side'] == 'Sell') or \
                            (orderid['price'] < bidprice and orderid['side'] == 'Buy'):
                        # ^ check if the order is closest
                        # move order up/down in the books to best execution

                        if orderid['side'] == "Sell":
                            bm.amend_order({'orderID': orderid['orderID'], 'price': askprice})
                            sleep(.01)
                        else:
                            bm.amend_order({'orderID': orderid['orderID'], 'price': bidprice})
                            sleep(.01)
                sleep(.01)
            if bm.open_orders() == []:
                orderid = None


if __name__ == '__main__':
    # plt_mash_and_predict(ns)
    main()
