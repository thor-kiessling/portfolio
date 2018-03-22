import numpy as np
import talib.stream as ta
import datetime
from talib import MA_Type
from scipy.optimize import differential_evolution
from matplotlib import pyplot as plt
import os
import sys
import math
from statistics import stdev as standard_deviation
import get_fund_history_bitmex
import get_history_bitmex
from market_maker import bitmex
import api_keys

os.nice(15)

import globalvar

plot_minusdi = []
plot_plusdi = []
long_enter_x = []
short_enter_x = []
long_enter_y = []
short_enter_y = []
long_exit_x = []
short_exit_x = []
long_exit_y = []
short_exit_y = []
return_graph_x = []
return_graph_y = []

API_KEY = api_keys.bmx_api_key
API_SECRET = api_keys.bmx_api_secret
bm = bitmex.BitMEX(base_url='https://www.bitmex.com/api/v1/', symbol='XBTUSD', login=None,
                       password=None, otpToken=None, apiKey=API_KEY,
                       apiSecret=API_SECRET, orderIDPrefix='jose')

print("Updating fund history")

while not get_fund_history_bitmex.main(bm):
    pass
while not get_history_bitmex.main(bm):
    pass

bm.exit()


def extract_data():
    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []
    print("loading from DB")
    globalvar.dbcur.execute("SELECT * FROM bmxswap ORDER BY time ASC")
    data = globalvar.dbcur.fetchall()
    # dt = np.dtype(np.double)
    print("converting from DB")
    data = np.array(data)
    # Iterate over the rows, splitting the label from the features.
    print("iterating over data")
    for index, line in enumerate(data):
        #if index > len(data) - 12000: continue  # TODO grabs final N minutes
        # print(len(temp_list))
        # print(len(fvecs))
        fvecs.append(line[:])

        # print(fvecs)
        if index % 1000 == 0:
            print(index / len(data))
    print("finalizing load")
    print(data.shape)
    print(np.array(fvecs).shape)
    return np.array(fvecs)


# class Position(object):
#
#     def __init__(self, start_btc=.1, maxout_percent=6):
#         self.in_position = False
#         self.maxout_percent = maxout_percent
#         self.btc = start_btc  # starting value of btc in accounts combined
#         self.long, self.short = None, None
#         self.margincalled = False
#
#     def enter_position(self, entry_price):
#         # entry price-0, usd contract amounts-1, btc value-2
#         self.long = [entry_price, entry_price * 95 * min(self.btc/2.0, 2), min(self.btc/2.0, 2)]
#         self.short = [entry_price, entry_price * 95 * min(self.btc/2.0, 2), min(self.btc/2.0, 2)]
#         self.btc -= self.long[2]
#         self.btc -= self.short[2]
#         self.margincalled = False
#         self.in_position = True
#
#     def exit_position(self, exit_price):
#         if self.long is None:
#             return self.btc
#         if not self.margincalled:
#             self.check_margin_call(price=exit_price)
#         if self.long[1] != 0:
#             self.long[2] += ((exit_price - self.long[0]) / self.long[0]) * self.long[1] / exit_price
#         if self.short[1] != 0:
#             self.short[2] += ((self.short[0] - exit_price) / self.short[0]) * self.short[1] / exit_price
#         self.btc = self.long[2] + self.short[2] + self.btc
#         self.btc *= .99
#
#         self.long, self.short = None, None
#         self.in_position = False
#
#                 #              price diff              price diff %     * amount wagered / usd price to get btc
#         return self.btc
#
#     def check_margin_call(self, price):
#         if self.margincalled and self.long is None:
#             return self.margincalled
#         if ((price - self.long[0]) / self.long[0]) * self.long[1] / price <= -self.long[2]:
#             self.long[2] = 0
#             self.long[1] = 0
#             self.margincalled = True
#         if ((self.short[0] - price) / self.short[0]) * self.short[1] / price <= -self.short[2]:
#             self.short[2] = 0
#             self.short[1] = 0
#             self.margincalled = True
#         if self.margincalled and self.in_position and self.long is not None and self.short is not None:
#             if self.short[1] == 0:
#                 if  ((price - self.long[0]) / self.long[0]) * self.long[1] / price <= -self.long[2]*.5:
#                 #                  if profit                 is less than            50% btc amount
#                     self.exit_position(price)
#                     return
#                 elif((price - self.long[0]) / self.long[0]) * self.long[1] / price >= self.long[2] * self.maxout_percent :
#                     #         if profit                                         is greater than     3x profit
#                     self.exit_position(price)
#                     return
#             if self.long[1] == 0:
#                 if ((self.short[0] - price) / self.short[0]) * self.short[1] / price <= -self.short[2]*.5:
#                     self.exit_position(price)
#                 elif ((self.short[0] - price) / self.short[0]) * self.short[1] / price >= self.short[2] * self.maxout_percent:
#                     self.exit_position(price)
#         return self.margincalled
        #if price >
#

# position_live = Position(start_btc=.1)
# position_live.enter_position(entry_price=5800)
# position_live.check_margin_call(price=5900)
# position_live.check_margin_call(price=5750)
# position_live.check_margin_call(price=5900)
# print(position_live.exit_position(exit_price=5850))
# position_live.enter_position(entry_price=5800)
# print(position_live.exit_position(exit_price=5850))
# position_live.enter_position(entry_price=5800)
# print(position_live.exit_position(exit_price=5820))
# position_live.enter_position(entry_price=5800)
# print(position_live.exit_position(exit_price=5820))
# position_live.enter_position(entry_price=5800)
# print(position_live.exit_position(exit_price=5820))
# position_live.enter_position(entry_price=5800)
# print(position_live.exit_position(exit_price=5950))
# position_live.enter_position(entry_price=5800)
# print(position_live.exit_position(exit_price=6050))
# position_live.enter_position(entry_price=6153)
# print(position_live.exit_position(exit_price=5791))
# raise Exception
def objective(variables, data=None, size=None, train=None, plot=None, try_leverages=None):
    variables[0]= int(variables[0])
    variables[1]= int(variables[1])
    if variables[0] < variables[1]: variables[1] = variables[0]
    longterm_ema_period = int(variables[2])
    longterm_ema = 0
    leverage = variables[3]
    usd = 1.0  # investing 5 dollars at 20x
    # position = []
    long = False
    in_position = False
    trades = 0
    EMA_length = int(variables[0]) + 1
    # EMA_length = max(EMA_length, 80000)
    ema1, ema2, adx, plusdi, minusdi, buyselllist= [], [], [100], [], [], []
    # plusdi_ema, minusdi_ema, adx_ema = 0,0,0
    returns = 1
    initialized = False
    max_draw_down = 0
    mdd_relative = 0
    peak_value = usd
    stoploss_triggers = 0
    reversal_flag = False
    leverage_flag = False
    local_return_graph_x = []
    local_return_graph_y = []
    longterm_history = []

    #if abs(variables[0]) > abs(variables[2]) + abs(variables[4]): return 0
    #if abs(variables[4]) > abs(variables[2]) + abs(variables[0]): return 0

    #variables[0], variables[2], variables[4] = 0,0,0\
    #ema1_ema, ema2_ema = 0,0

    for iteration in range(size):
        if iteration > .9 * size and train: break
        if iteration < .9 * size and not train: continue  # use 90% train, 10% test, separated in time
        line = data[-1 * size + iteration]  # single line
        if longterm_ema == 0:
            longterm_ema = line[4]
        # if iteration != size - 1:
        #     max_exec_price = np.amax(data[[[-1 * size + iteration],[-1*size+iteration+2]], [1,2,3,4]])
        #     min_exec_price = np.amin(data[[[-1 * size + iteration],[-1*size+iteration+2]], [1,2,3,4]])
        # else:
        #     max_exec_price = np.amax(line[1:5])
        #     min_exec_price = np.amin(line[1:5])
        min_exec_price, max_exec_price = np.mean(line[1:5]), np.mean(line[1:5])
        # for item in [plusdi_ema, minusdi_ema, adx_ema]:
        #     if item == 0: item = line[1]
        try:
            # multiplier = (2 / (variables[1] + 1))
            truncated_data = data[-1 * size + iteration - EMA_length:-1 * size + iteration, :]  # numpy array view
            # ema1.append(ta.TRIMA(truncated_data[:, 4], timeperiod=variables[0]))
            ema1.append(ta.EMA(truncated_data[:, 4], timeperiod=variables[0]))
            if len(ema1) < 2: continue
            ema2.append(ta.EMA(np.array(ema1[int(-variables[1]):]), timeperiod=int(min(len(ema1), variables[1]))))
            longterm_ema = (line[4] - longterm_ema) * (2 / (longterm_ema_period + 1)) + longterm_ema
            if plot:
                longterm_history.append(longterm_ema)
            # ema1.append(ta.ULTOSC(truncated_data[:, 2], truncated_data[:, 3], truncated_data[:, 4],
            #                      timeperiod1=int(variables[0]/4.0), timeperiod2=int(variables[0]/2.0), timeperiod3=int(variables[0])))
            # adx_close = ta.ADX(truncated_data[:, 2], truncated_data[:, 3], truncated_data[:, 4], timeperiod=variables[2])/(100/variables[2])
            # if len(adx) == 0: adx = [adx_close]
            # else:
            #     adx.append((adx_close - adx[-1]) * multiplier + adx[-1])
            # plusdi_close = ta.PLUS_DI(truncated_data[:, 2], truncated_data[:, 3], truncated_data[:, 4], timeperiod=variables[2])/(100/variables[2])
            # if len(plusdi) == 0: plusdi = [plusdi_close]
            # else:
            #     plusdi.append((plusdi_close - plusdi[-1]) * multiplier + plusdi[-1])
            # minusdi_close = ta.MINUS_DI(truncated_data[:, 2], truncated_data[:, 3], truncated_data[:, 4], timeperiod=variables[2])/(100/variables[2])
            # if len(minusdi) == 0: minusdi = [minusdi_close]
            # else:
            #     minusdi.append((minusdi_close - minusdi[-1]) * multiplier + minusdi[-1])
            #print(int((variables[1]/stddev[-1])*.0056), stddev[-1])
            # ema2.append(ta.EMA(truncated_data[:, 4], timeperiod=variables[1]))  # todo change
            # ema1_last = ema1[-1]
            # ema2_last = ema2[-1]
            # ATR = ta.ATR(truncated_data[:, 2], truncated_data[:, 3], truncated_data[:, 4], timeperiod=variables[1]) / line[4]
            # uband, middleband, lband = ta.BBANDS(truncated_data[:, 4], timeperiod=variables[1], matype=5)
            # bw = (uband - lband) / middleband
            # adx = ta.ADX(truncated_data[:, 2], truncated_data[:, 3], truncated_data[:, 4], timeperiod=variables[1])/100
            # rsi = ta.RSI(truncated_data[:, 4], timeperiod=variables[1])/100.0
            # mfi = ta.MFI(truncated_data[:, 2], truncated_data[:, 3], truncated_data[:, 4], truncated_data[:, 5], timeperiod=variables[1])
            # if math.isnan(adx[-1]) or math.isnan(ema1[-1]) or math.isnan(ema2[-1]):
            #     raise ValueError
        except:
            print(list(variables), EMA_length)
            #print(ema1, ema2, adx)
            e = sys.exc_info()
            print(("<p>Error: %s</p>" % e))
        # multiplier = (2 / (10 + 1))
        # plusdi_ema = (plusdi[-1] - plusdi_ema) * multiplier + plusdi_ema
        # minusdi_ema = (minusdi[-1] - minusdi_ema) * multiplier + minusdi_ema
        # plusdi_ema = (adx[-1] - adx_ema) * multiplier + adx_ema

        if (iteration <= 1 and train) or (iteration <= .9 * size + 1 and not train): continue
        # truncated_data = data[-1 * size + iteration - EMA_length:-1 * size + iteration, :]  # numpy array view

        buy = 0
        sell = 0
        # if ema1[-1] > line[1]:
        #     buy += variables[0] #* (ema1[-1] - line[1])
        # else:
        #     sell += variables[0] #* (ema1[-1] - line[1])
        # if ema2[-1] > line[1]:
        #     buy += variables[2] #* (ema2[-1] - line[1])
        # else:
        #     sell += variables[2] #* (ema2[-1] - line[1])

        #if abs(ema1[-1] - ema2[-1])/ema2[-1] > variables[2]:
        #if plusdi[-1] > minusdi[-1]:
        #slope = ema1[-1] - ema1[-2]
        if ema2[-1] > ema2[-2]:
            buy += 1
        else:
            sell += 1

        buy_sell = buy - sell
        buyselllist.append(buy_sell)
        #print(buy_sell)  #
        # end programmable bits, begin simulation
        if not initialized:
            initialized = True
            # position_live = Position(start_btc=usd/line[4])
            # in_position = False
            # returns = position_live.btc * line[4]
            # peak_value = returns

            if buy_sell > 0:  # only executed on first run, from dollars to btc/shorts
                position = [usd * .9997, max_exec_price]  # store entry price
                usd = 0
                long = True
                # in_position = True
                last_trade_time = iteration
                if plot:
                   #global long_enter_x
                   #global long_enter_y
                    long_enter_x.append(line[0])
                    long_enter_y.append(line[1]-.1)
                    return_graph_x.append(line[0])
                    return_graph_y.append(position[0])
                    local_return_graph_x.append(line[0])
                    local_return_graph_y.append(position[0])
                    init_trade = "long"
                continue
            elif buy_sell < 0:
                position = [usd * .9997, min_exec_price]  # store entry price
                usd = 0
                long = False
                # in_position = True
                last_trade_time = iteration
                if plot:
                   #global short_enter_x
                   #global short_enter_y
                    short_enter_x.append(line[0])
                    short_enter_y.append(line[1]+.1)
                    return_graph_x.append(line[0])
                    return_graph_y.append(position[0])
                    local_return_graph_x.append(line[0])
                    local_return_graph_y.append(position[0])
                    init_trade = "short"
                continue

        # time_since_last_trade = iteration - last_trade_time
        if in_position and position[0] > peak_value:
            peak_value = position[0]
        if initialized:
            if long:
                returns = usd + (line[1] / position[1]) * position[0]
            if not long:
                returns = usd + (position[1] /line[1]) * position[0]
            draw_down = (returns - peak_value) / peak_value
            if draw_down < max_draw_down:
                max_draw_down = draw_down
                #if plot: print("new mdd long from", position[1], "to", line[1])
            dd_relative = draw_down
            # trailing_stop_max = (line[1] - np.max(data[-size + last_trade_time:-size + iteration, 4])) / np.max(
            #     data[-size + last_trade_time:-size + iteration, 4])
            if dd_relative < mdd_relative:
                mdd_relative = dd_relative
                    # if plot: print("new relative dd long from", position[1], "to", line[1], "at", iteration)
            #

            #     draw_down = (returns - peak_value) / peak_value
            #     if draw_down < max_draw_down:
            #         max_draw_down = draw_down
            #         #if plot: print("new mdd short from", position[1], "to", line[1])
            #     dd_relative = (returns - position[0]) / position[0]
            #     trailing_stop_max = (np.min(data[-size + last_trade_time:-size + iteration, 4]) - line[1]) / line[1]
            #
            #     if dd_relative < mdd_relative:
            #         mdd_relative = dd_relative
            #         # if plot: print("new rel dd short from", position[1], "to", line[1], "at", iteration)
        # if not in_position:
        #     returns = usd
        #     if long:
        #         trailing_stop_max = (line[1] - np.min(data[-size + last_trade_time:-size + iteration, 4])) / np.min(data[-size + last_trade_time:-size + iteration, 4])
        #     else:
        #         trailing_stop_max = (np.max(data[-size + last_trade_time:-size + iteration, 4]) - line[1]) / line[1]
        # if in_position:
        #     if long:
        #         # if adx[-1] > variables[3]:# and line[1] > ema2[-1] - (ema2[-1] * variables[4]):  # trailing_stop_max > variables[4]:
        #             desire_position = True
        #         # else:
        #         #     desire_position = False
        #     else:
        #         # if adx[-1] > variables[3]:# and line[1] < ema2[-1] + (ema2[-1] * variables[4]):
        #             desire_position = True
        #         # else:
        #         #     desire_position = False
        # if not in_position:
        #     if long:
        #         # if adx[-1] > variables[3]:# and line[1] > ema2[-1]: # and trailing_stop_max > -variables[4] * 4:
        #             desire_position = True
        #         # else:
        #         #     desire_position = False
        #     else:
        #         # if adx[-1] > variables[3]:# and line[1] < ema2[-1]:
        #             desire_position = True
        #         # else:
        #         #     desire_position = False

        # if not desire_position and in_position:  # stop loss section 1
        #     if not long:
        #         usd = (position[1] / max_exec_price) * position[0]
        #         position = 0
        #         # position = [usd * .9997, max(line[1:5])]
        #         # usd = 0.0
        #         #long = False
        #         in_position = False
        #         stoploss_triggers += 1
        #         last_trade_time = iteration
        #         reversal_flag = True
        #         if plot:
        #            #global short_exit_x
        #            #global short_exit_y
        #             short_exit_x.append(line[0])
        #             short_exit_y.append(line[1] + .1)
        #             return_graph_x.append(line[0])
        #             return_graph_y.append(returns)
        #             local_return_graph_x.append(line[0])
        #             local_return_graph_y.append(returns)
        #         continue
        #     if long:  # stoploss part 1
        #         usd = (min_exec_price / position[1]) * position[0]  # assume bad execution
        #         position = 0
        #         # position = [usd * .9997, min(line[1:5])]
        #         # usd = 0.0
        #         #long = True
        #         in_position = False
        #         stoploss_triggers += 1
        #         last_trade_time = iteration
        #         reversal_flag = True
        #         if plot:
        #             #global long_exit_x
        #             #global long_exit_y
        #             long_exit_x.append(line[0])
        #             long_exit_y.append(line[1] - .1)
        #             return_graph_x.append(line[0])
        #             return_graph_y.append(returns)
        #             local_return_graph_x.append(line[0])
        #             local_return_graph_y.append(returns)
        #         continue
        # upper_bound = ema2[-1] * (1+(variables[2]*(variables[3]-(min(time_since_last_trade, variables[4])/(variables[4]/(variables[3]-1))))))
        # lower_bound = ema2[-1] * (1-(variables[2]*(variables[3]-(min(time_since_last_trade, variables[4])/(variables[4]/(variables[3]-1))))))


        # if desire_position and in_position: # section 2
        #     if long:  # section where we release control back to buy_sell
        #         if line[1] < lower_bound:
        #             reversal_flag = False
        #
        #     else:
        #         if line[1] > upper_bound:
        #             reversal_flag = False
        #
        #     if long:
        #         if buy_sell > 0.0:
        #             reversal_flag = False
        #
        #     else:
        #         if buy_sell < 0.0:
        #             reversal_flag = False
        # in_position = position_live.in_position
        # if in_position:
        #     position_live.check_margin_call(line[4])
        #     in_position = position_live.in_position
        #     if not in_position:
        #         desire_position = False
        if buy_sell > 0 and not long and initialized:  # main buy/sell todo modify this
            if leverage_flag:
                delta = (((position[1] / max_exec_price) - 1) * leverage) + 1
                usd = delta * position[0]
                leverage_flag = False
            else:
                usd = (position[1] / max_exec_price) * position[0]
            position = 0
            position = [usd * .9997, max_exec_price]
            usd = 0.0
            long = True
            if longterm_ema > line[4]:
                leverage_flag = True
            # in_position = True
            trades += 1
            last_trade_time = iteration
            # if line[1] > upper_bound:
            #     reversal_flag = True
            #     stoploss_triggers += 1
            # else:
            #     reversal_flag = False
            if plot:
                #global long_enter_x
                #global long_enter_y
                long_enter_x.append(line[0])
                long_enter_y.append(line[4])
                return_graph_x.append(line[0])
                return_graph_y.append(returns)
                local_return_graph_x.append(line[0])
                local_return_graph_y.append(returns)
            continue

        if buy_sell < 0 and long and initialized:  #  main buy/sell 2

            #                      on 20x
            if leverage_flag:
                delta = (((min_exec_price / position[1]) - 1) * leverage ) +1
                usd =  delta * position[0]
                leverage_flag = False
            else:
                usd = (min_exec_price / position[1]) * position[0]  # assume bad execution
            position = 0
            position = [usd * .9997, min_exec_price]
            usd = 0.0
            long = False
            # in_position = True
            if longterm_ema < line[4]:
                leverage_flag = True
            trades += 1
            last_trade_time = iteration
            # if line[1] < lower_bound:
            #     reversal_flag = True
            #     stoploss_triggers += 1
            # else:
            #     reversal_flag = False
            if plot:
               #global short_enter_x
               #global short_enter_y
                short_enter_x.append(line[0])
                short_enter_y.append(line[4])
                return_graph_x.append(line[0])
                return_graph_y.append(returns)
                local_return_graph_x.append(line[0])
                local_return_graph_y.append(returns)
            continue

        # if not desire_position and not in_position:  # section 3
        #     if not long and ((buy_sell > 0.0) ^ (line[1] > upper_bound)): # moves understanding of long/short desire despite not being in position
        #         #usd = (position[1] / max(line[1:5])) * position[0]
        #         #position = 0
        #         #position = [usd * .9997, max(line[1:5])]
        #         #usd = 0.0
        #         long = True
        #         #in_position = True
        #         #trades += 1
        #         last_trade_time = iteration
        #         if line[1] > ema2[-1] * (1 + variables[2]):
        #             reversal_flag = True
        #         else:
        #             reversal_flag = False
        #         # if plot:
        #         #    #global long_enter_x
        #         #    #global long_enter_y
        #         #     long_enter_x.append(line[0])
        #         #     long_enter_y.append(line[1]-.1)
        #         #     return_graph_x.append(line[0])
        #         #     return_graph_y.append(returns)
        #         continue
        #
        #     if long and ((buy_sell < 0.0) ^ (line[1] < lower_bound)):  # moves understanding of long/short desire despite not being in position
        #         #usd = (min(line[1:5]) / position[1]) * position[0]
        #         #position = 0
        #         # position = [usd * .9997, min(line[1:5])]
        #         # usd = 0.0
        #         long = False
        #         # in_position = True
        #         # trades += 1
        #         last_trade_time = iteration
        #         if line[1] < lower_bound:
        #             reversal_flag = True
        #         else:
        #             reversal_flag = False
        #         # if plot:
        #         #    #global short_enter_x
        #         #    #global short_enter_y
        #         #     short_enter_x.append(line[0])
        #         #     short_enter_y.append(line[1]+.1)
        #         #     return_graph_x.append(line[0])
        #         #     return_graph_y.append(returns)
        #         continue


        # if desire_position and not in_position:  # if adx reverses direction, reenter position section 4
        #     if not long:  #
        #         #usd = (position[1] / max(line[1:5])) * position[0]
        #         #position = 0
        #         position = [usd * .9997, min_exec_price]
        #         usd = 0.0
        #         #long = True
        #         in_position = True
        #         trades += 1
        #         last_trade_time = iteration
        #         reversal_flag = False
        #         if plot:
        #            #global short_enter_x
        #            #global short_enter_y
        #             short_enter_x.append(line[0])
        #             short_enter_y.append(line[1]+.1)
        #             return_graph_x.append(line[0])
        #             return_graph_y.append(returns)
        #             local_return_graph_x.append(line[0])
        #             local_return_graph_y.append(returns)
        #         continue
        #
        #     if long:
        #         #usd = (min(line[1:5]) / position[1]) * position[0]  # assume bad execution
        #         #position = 0
        #         position = [usd * .9997, min_exec_price]
        #         usd = 0.0
        #         #long = False
        #         in_position = True
        #         trades += 1
        #         last_trade_time = iteration
        #         reversal_flag = False
        #         if plot:
        #            #global long_enter_x
        #            #global long_enter_y
        #             long_enter_x.append(line[0])
        #             long_enter_y.append(line[1] - .1)
        #             return_graph_x.append(line[0])
        #             return_graph_y.append(returns)
        #             local_return_graph_x.append(line[0])
        #             local_return_graph_y.append(returns)
        #         continue

        if returns < 50 and not plot:  # fail-quick
            returns = 0
            position = [0, line[1]]
            max_draw_down = -.9271
            break
    # ema1_switches = 0
    # for index, item in enumerate(ema1):
    #     if (item > ema1[index-1]) is not (ema1[index-1] > ema1[index-2]):
    #         ema1_switches += 1
    # ema2_switches = 0
    # for index, item in enumerate(ema2):
    #     if (item > ema2[index - 1]) is not (ema2[index - 1] > ema2[index - 2]):
    #         ema2_switches += 1
    # macd_switches = 0
    # for index, item in enumerate(buyselllist):
    #     if (item > 0) is not (buyselllist[index-1] > 0):
    #         macd_switches += 1
    # buyselllist = []
    # longest_time_since_last_high = 0
    # for index, tick in enumerate(local_return_graph_y):  # TODO TEMPORARY
    #     if index == 0:
    #         buyselllist.append(0)
    #         continue
    #     #print((tick - local_return_graph_y[index - 1]) / tick)
    #     buyselllist.append((tick - local_return_graph_y[index - 1]) / tick)
    # print(sorted(buyselllist))
    # if plot:
        # for leverage in range(1, 101):
        #     max_value = 0
        #     last_value = 0
        #     time_of_last_high = local_return_graph_x[0]
        #     time_since_last_high = 0
        #
        #     for tickindex, tick in enumerate(local_return_graph_y):
        #         if tickindex == 0:
        #             max_value = tick
        #             last_value = tick
        #             continue
        #         delta = (tick - local_return_graph_y[tickindex - 1]) / tick
        #         delta *= leverage
        #         last_value *= delta + 1
        #         if last_value > max_value:
        #             time_since_last_high = local_return_graph_x[tickindex] - time_of_last_high
        #             time_of_last_high = local_return_graph_x[tickindex]
        #             # if time_since_last_high > longest_time_since_last_high:
        #             #     longest_time_since_last_high = time_since_last_high
        #             max_value = last_value
        #         if last_value < max_value / float(2):
        #             break
        #     if last_value < max_value / float(2):  # when broken out of the inner for by too much leverage
        #         max_leverage = leverage - 1
        #         break
        #     elif leverage == 100:
        #         print("got max leverage 100 somehow")
        #         max_leverage = leverage
        #         break
        # best_leverage = [0,0]
        # best_value = 0
        # for short_leverage in range(0,101):
        #     for long_leverage in range(0,101):
        #         max_value = 0
        #         last_value = 0
        #
        #         for tickindex, tick in enumerate(local_return_graph_y):
        #             if tickindex == 0:
        #                 max_value = tick
        #                 last_value = tick
        #                 continue
        #             delta = (tick - local_return_graph_y[tickindex - 1]) / tick
        #             if (init_trade == "short") ^ tickindex % 2 == 0:
        #                 delta *= short_leverage
        #                 short_fee = 1 #- .0000025 * short_leverage
        #                 last_value *= delta + short_fee
        #
        #             else:
        #                 delta *= long_leverage
        #                 long_fee = 1 #- .0000025 * long_leverage
        #                 last_value *= delta + long_fee
        #
        #             if last_value > max_value:
        #                 # if time_since_last_high > longest_time_since_last_high:
        #                 #     longest_time_since_last_high = time_since_last_high
        #                 max_value = last_value
        #             if last_value < max_value / float(4):
        #                 last_value = 0
        #                 break
        #         if last_value > best_value:
        #             best_leverage = [long_leverage, short_leverage]
        #             best_value = last_value
        # print("best long/short leverage", best_leverage, best_value)
    # if try_leverages:
    #     long_leverage = try_leverages[0]
    #     short_leverage = try_leverages[1]
    #     max_value = 0
    #     last_value = 0
    #
    #     for tickindex, tick in enumerate(local_return_graph_y):
    #         if tickindex == 0:
    #             max_value = tick
    #             last_value = tick
    #             continue
    #         delta = (tick - local_return_graph_y[tickindex - 1]) / tick
    #         if (init_trade == "short") ^ tickindex % 2 == 0:
    #             delta *= short_leverage
    #             short_fee = 1  # - .0000025 * short_leverage
    #             last_value *= delta + short_fee
    #
    #         else:
    #             delta *= long_leverage
    #             long_fee = 1  # - .0000025 * long_leverage
    #             last_value *= delta + long_fee
    #
    #         if last_value > max_value:
    #             # if time_since_last_high > longest_time_since_last_high:
    #             #     longest_time_since_last_high = time_since_last_high
    #             max_value = last_value
    #     print("for try_leverage", try_leverages, last_value)

    # max_leverage = min(math.floor(-1/min(mdd_relative, -.0001)), max_leverage)



    # print(max(adx), "avg:", np.average(adx), min(adx))
    if plot:
        print("MDD%", max_draw_down * 100, "dd_rel%:",
               mdd_relative * 100, "max leverage:")
    else:
        print("MDD%", max_draw_down * 100, "dd_rel%:",
              mdd_relative * 100)
    #print(max(atr), min(atr))
    #print("min, max, stddev, avg of stddev", min(stddev), max(stddev), np.std(stddev), np.average(stddev))


    if plot:
        # global plot_minusdi
        # global plot_plusdi
        if train:
            # plt.plot(data[int(-len(ema1)-(size/10)):int(-(size/10)), 0], ema1)
            plt.plot(data[int(-len(ema2)-(size/10)):int(-(size/10)), 0], ema2)
            plt.plot(data[int(-len(ema2)-(size/10)):int(-(size/10)), 0], longterm_history, 'b-')
            ##global plot_adx

            # plot_adx += adx
            # plot_minusdi += minusdi
            # plot_plusdi += plusdi
        else:
            # plt.plot(data[-len(ema1):, 0], ema1)
            plt.plot(data[-len(ema2):, 0], ema2)
            plt.plot(data[-len(longterm_history):, 0], longterm_history)
            # global plot_adx

            # plot_adx += adx
            # plot_minusdi += minusdi
            # plot_plusdi += plusdi
    if long:
        returns = usd + (min_exec_price / position[1]) * position[0]
        if plot:
            return_graph_x.append(line[0])
            return_graph_y.append(returns)
            local_return_graph_x.append(line[0])
            local_return_graph_y.append(returns)
        print(list(variables), returns, trades)
        if plot:
            print("Score metric:", returns * (1 + max_draw_down))
        else:
            print("Score metric:", returns * (1 + max_draw_down))
    else:
        returns = usd + (position[1] / max_exec_price) * position[0]
        if plot:
            return_graph_x.append(line[0])
            return_graph_y.append(returns)
            local_return_graph_x.append(line[0])
            local_return_graph_y.append(returns)
            if trades < 10 and not plot and train: returns = 0
        print(list(variables), returns, trades)
        if plot:
            print("Score metric:", returns * (1 + max_draw_down))
        else:
            print("Score metric:", returns * (1 + max_draw_down))

    return -(returns * (1 + max_draw_down))
# plot_adx = []

def main():
    data = extract_data()
    minutes_train = 510000  # four zeros = 1 week, five zeroes = 10 weeks,  1300000 = all my data 500000=1year
    #bounds = [(-0, 0), (2, 1000), (-0, 0), (40, 40000), (-1, 1), (10, 40000), (.0001, .002), (10, 40000)]
    #                                                                                atr reverse trigger, lookback 6/7
    #                      max            max        max         max
    #bounds = [(0.0, .50), (10, 100)]
    #           diff%,     ATR timerange
    #              0            1
    bounds = [(10, 2000), (10, 1000),   (1000, 100000), (1, 10)]
    #          ema 1            ema2   longterm ema     leverage
    # document for bounds
    fig, ax1 = plt.subplots()
    # result = differential_evolution(objective, bounds, args=(data, minutes_train, True, False), disp=True,
    #                              maxiter=80, popsize=10, callback=callback, strategy='best2bin', polish=False) # todo paralleled popsize was 3
    # print(list(result.x), -result.fun)
    # objective(result.x, data=data, size=minutes_train, train=True, plot=True)
    # test_result = objective(result.x, data=data, size=minutes_train, train=False, plot=True)

    # objective([0.0098357358852500382, 31.0], data=data, size=minutes_train, train=True, plot=True) # best of inverse %b
    # test_result = objective([0.0098357358852500382, 31.0], data=data, size=minutes_train, train=False, plot=True)
    # it resulted in 604k, 2431, and 6/8 post-split bankruptcies
    # objective([0.8139530939701376, 94.0], data=data, size=minutes_train, train=True, plot=True) # best of rsi
    # test_result = objective([0.8139530939701376, 94.0], data=data, size=minutes_train, train=False, plot=True)
    # it resulted in
    # objective([7.8045130486381631e-05, 3567.0], data=data, size=minutes_train, train=True, plot=True)  # best of ema slope
    # test_result = objective([7.8045130486381631e-05, 3567.0], data=data, size=minutes_train, train=False, plot=True)
    # it resulted in

    # objective([30.0], data=data, size=minutes_train, train=True, plot=True)  # best of TRIMA
    # test_result = objective([30.0], data=data, size=minutes_train, train=False, plot=True)
    # 1656 resulted in 860 returns and negative new returns
    # objective([4000.0], data=data, size=minutes_train, train=True, plot=True)  # best of smoothEMA
    # test_result = objective([4000.0], data=data, size=minutes_train, train=False, plot=True)
    # 829 resulted in 716 returns and negative new returns
    # DING DING BIG WINNER USE THIS
    # objective([1731.0, 658.0], data=data, size=minutes_train, train=True, plot=True)  # best of variablysmoothEMA
    # test_result = objective([1731.0, 658.0], data=data, size=minutes_train, train=False, plot=True)
    # this resulted in 1400 returns and 160 new returns 500k
    # shorty winner
    objective([1054.0, 373.0, 50000, 5], data=data, size=minutes_train, train=True, plot=True)  # best of shorter doubleEMA
    test_result = objective([1054.0, 373.0, 50000, 5], data=data, size=minutes_train, train=False, plot=True)





    plt.plot(data[-minutes_train:, 0], data[-minutes_train:, 4],  'k-')

    plt.plot(long_enter_x, long_enter_y, 'g^')
    plt.plot(short_enter_x, short_enter_y, 'rv')
    plt.plot(long_exit_x, long_exit_y, 'gv')
    plt.plot(short_exit_x, short_exit_y, 'r^')
    ax2 = ax1.twinx()

    ax2.plot(return_graph_x, return_graph_y)
    ax2.set_yscale('log')
    # print(len(plot_adx))
    # print(max(plot_adx), min(plot_adx))
    # ax2.plot(data[-minutes_train-1:, 0], plot_adx)
    # ax2.plot(data[-minutes_train-1:, 0], plot_minusdi)
    # ax2.plot(data[-minutes_train-1:, 0], plot_plusdi)
    fig.tight_layout()
    plt.tight_layout()
    # for index in range(len(return_graph_x)): todo only for exporting to excel to compare to real returns
    #     print(str(datetime.datetime.fromtimestamp(return_graph_x[index]))+","+str(return_graph_y[index]))
    plt.show()

def callback(xk, convergence=None):
    print("generation best:", list(xk), convergence)
    return


if __name__ == '__main__':
    main()
