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

def extract_data():
    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []
    print("loading from DB")
    globalvar.dbcur.execute("SELECT * FROM futures ORDER BY time ASC")
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


def objective(variables, data=None, size=None, train=None, plot=None):
    variables[0], variables[1] = \
        int(variables[0]), int(variables[1])
    usd = 100.0  # investing 5 dollars at 20x
    position = []
    long = False
    in_position = False
    trades = 0
    EMA_length = max(int(variables[0]), int(variables[1]))*3
    ema1, ema2, adx, plusdi, minusdi, buyselllist= [], [], [100], [], [], []
    # plusdi_ema, minusdi_ema, adx_ema = 0,0,0
    returns = 100
    initialized = False
    max_draw_down = 0
    mdd_relative = 0
    peak_value = 0
    stoploss_triggers = 0
    reversal_flag = False
    local_return_graph_x = []
    local_return_graph_y = []

    #if abs(variables[0]) > abs(variables[2]) + abs(variables[4]): return 0
    #if abs(variables[4]) > abs(variables[2]) + abs(variables[0]): return 0

    #variables[0], variables[2], variables[4] = 0,0,0\
    #ema1_ema, ema2_ema = 0,0

    for iteration in range(size):
        if iteration > .9 * size and train: break
        if iteration < .9 * size and not train: continue  # use 90% train, 10% test, separated in time
        line = data[-1 * size + iteration]  # single line
        # for item in [plusdi_ema, minusdi_ema, adx_ema]:
        #     if item == 0: item = line[4]
        try:
            # multiplier = (2 / (variables[1] + 1))
            truncated_data = data[-1 * size + iteration - EMA_length:-1 * size + iteration, :]  # numpy array view
            ema1.append(ta.TEMA(truncated_data[:, 4], timeperiod=variables[0]))
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
            ema2.append(ta.EMA(truncated_data[:, 4], timeperiod=variables[1]))  # todo change
            ema1_last = ema1[-1]
            ema2_last = ema2[-1]
            if math.isnan(adx[-1]) or math.isnan(ema1[-1]) or math.isnan(ema2[-1]):
                raise ValueError
        except:
            print(list(variables), EMA_length)
            print(ema1, ema2, adx)
            e = sys.exc_info()
            print(("<p>Error: %s</p>" % e))
        # multiplier = (2 / (10 + 1))
        # plusdi_ema = (plusdi[-1] - plusdi_ema) * multiplier + plusdi_ema
        # minusdi_ema = (minusdi[-1] - minusdi_ema) * multiplier + minusdi_ema
        # plusdi_ema = (adx[-1] - adx_ema) * multiplier + adx_ema

        if (iteration <= 1 and train) or (iteration <= .9 * size + 1 and not train): continue
        truncated_data = data[-1 * size + iteration - EMA_length:-1 * size + iteration, :]  # numpy array view

        buy = 0
        sell = 0
        # if ema1[-1] > line[4]:
        #     buy += variables[0] #* (ema1[-1] - line[4])
        # else:
        #     sell += variables[0] #* (ema1[-1] - line[4])
        # if ema2[-1] > line[4]:
        #     buy += variables[2] #* (ema2[-1] - line[4])
        # else:
        #     sell += variables[2] #* (ema2[-1] - line[4])

        if ema1[-1] > ema2[-1]:
        #if plusdi[-1] > minusdi[-1]:
            buy += 1
        else:
            sell += 1

        buy_sell = buy - sell
        buyselllist.append(buy_sell)
        #print(buy_sell)  # todo debug
        # end programmable bits, begin simulation
        if not initialized:
            initialized = True
            if buy_sell > 0:  # only executed on first run, from dollars to btc/shorts
                position = [usd * .9997, np.max((line[1],line[4]))]  # store entry price
                usd = 0
                long = True
                in_position = True
                last_trade_time = iteration
                if plot:
                   #global long_enter_x
                   #global long_enter_y
                    long_enter_x.append(line[0])
                    long_enter_y.append(line[4]-.1)
                    return_graph_x.append(line[0])
                    return_graph_y.append(position[0])
                    local_return_graph_x.append(line[0])
                    local_return_graph_y.append(position[0])
                continue
            elif buy_sell < 0:
                position = [usd * .9997, np.min((line[1],line[4]))]  # store entry price
                usd = 0
                long = False
                in_position = True
                last_trade_time = iteration
                if plot:
                   #global short_enter_x
                   #global short_enter_y
                    short_enter_x.append(line[0])
                    short_enter_y.append(line[4]+.1)
                    return_graph_x.append(line[0])
                    return_graph_y.append(position[0])
                    local_return_graph_x.append(line[0])
                    local_return_graph_y.append(position[0])
                continue
            else:
                if buy_sell > 0:
                    long = True
                    in_position = False
                    last_trade_time = iteration
                    trailing_stop_max = 0
                    continue
                if buy_sell < 0:
                    long = False
                    in_position = False
                    last_trade_time = iteration
                    trailing_stop_max = 0
                    continue
        time_since_last_trade = iteration - last_trade_time
        if in_position and position[0] > peak_value:
            peak_value = position[0]
        if initialized and in_position:
            if long:
                returns = usd + (line[4] / position[1]) * position[0]
                draw_down = (returns - peak_value) / peak_value
                if draw_down < max_draw_down:
                    max_draw_down = draw_down
                    #if plot: print("new mdd long from", position[1], "to", line[4])
                dd_relative = (returns - position[0]) / position[0]
                trailing_stop_max = (line[4] - np.max(data[-size + last_trade_time:-size + iteration, 4])) / np.max(
                    data[-size + last_trade_time:-size + iteration, 4])
                if dd_relative < mdd_relative:
                    mdd_relative = dd_relative
                    # if plot: print("new relative dd long from", position[1], "to", line[4], "at", iteration)

            else:
                returns = usd + (position[1] /line[4]) * position[0]
                draw_down = (returns - peak_value) / peak_value
                if draw_down < max_draw_down:
                    max_draw_down = draw_down
                    #if plot: print("new mdd short from", position[1], "to", line[4])
                dd_relative = (returns - position[0]) / position[0]
                trailing_stop_max = (np.min(data[-size + last_trade_time:-size + iteration, 4]) - line[4]) / line[4]

                if dd_relative < mdd_relative:
                    mdd_relative = dd_relative
                    # if plot: print("new rel dd short from", position[1], "to", line[4], "at", iteration)
        if not in_position:
            returns = usd
            if long:
                trailing_stop_max = (line[4] - np.min(data[-size + last_trade_time:-size + iteration, 4])) / np.min(data[-size + last_trade_time:-size + iteration, 4])
            else:
                trailing_stop_max = (np.max(data[-size + last_trade_time:-size + iteration, 4]) - line[4]) / line[4]
        if in_position:
            if long:
                # if adx[-1] > variables[3]:# and line[4] > ema2[-1] - (ema2[-1] * variables[4]):  # trailing_stop_max > variables[4]:
                    desire_position = True
                # else:
                #     desire_position = False
            else:
                # if adx[-1] > variables[3]:# and line[4] < ema2[-1] + (ema2[-1] * variables[4]):
                    desire_position = True
                # else:
                #     desire_position = False
        if not in_position:
            if long:
                # if adx[-1] > variables[3]:# and line[4] > ema2[-1]: # and trailing_stop_max > -variables[4] * 4:
                    desire_position = True
                # else:
                #     desire_position = False
            else:
                # if adx[-1] > variables[3]:# and line[4] < ema2[-1]:
                    desire_position = True
                # else:
                #     desire_position = False

        if not desire_position and in_position:  # stop loss section 1
            if not long:
                usd = (position[1] / np.max((line[1],line[4]))) * position[0]
                position = 0
                # position = [usd * .9997, max(line[1:5])]
                # usd = 0.0
                #long = False
                in_position = False
                stoploss_triggers += 1
                last_trade_time = iteration
                reversal_flag = True
                if plot:
                   #global short_exit_x
                   #global short_exit_y
                    short_exit_x.append(line[0])
                    short_exit_y.append(line[4] + .1)
                    return_graph_x.append(line[0])
                    return_graph_y.append(returns)
                    local_return_graph_x.append(line[0])
                    local_return_graph_y.append(returns)
                continue
            if long:  # stoploss part 1
                usd = (np.min((line[1],line[4])) / position[1]) * position[0]  # assume bad execution
                position = 0
                # position = [usd * .9997, min(line[1:5])]
                # usd = 0.0
                #long = True
                in_position = False
                stoploss_triggers += 1
                last_trade_time = iteration
                reversal_flag = True
                if plot:
                    #global long_exit_x
                    #global long_exit_y
                    long_exit_x.append(line[0])
                    long_exit_y.append(line[4] - .1)
                    return_graph_x.append(line[0])
                    return_graph_y.append(returns)
                    local_return_graph_x.append(line[0])
                    local_return_graph_y.append(returns)
                continue
        upper_bound = ema2[-1] * (1+(variables[2]*(variables[3]-(min(time_since_last_trade, variables[4])/(variables[4]/(variables[3]-1))))))
        lower_bound = ema2[-1] * (1-(variables[2]*(variables[3]-(min(time_since_last_trade, variables[4])/(variables[4]/(variables[3]-1))))))


        if desire_position and in_position: # section 2
            if long:
                if line[4] < lower_bound:
                    reversal_flag = False

            else:
                if line[4] > upper_bound:
                    reversal_flag = False


            if not long and ((buy_sell > 0.0) or (line[4] > upper_bound)) and not reversal_flag: # main buy/sell
                usd = (position[1] / np.max((line[1],line[4]))) * position[0]
                position = 0
                position = [usd * .9997, np.max((line[1],line[4]))]
                usd = 0.0
                long = True
                in_position = True
                trades += 1
                last_trade_time = iteration
                if line[4] > upper_bound:
                    reversal_flag = True
                    stoploss_triggers += 1
                else:
                    reversal_flag = False
                if plot:
                    #global long_enter_x
                    #global long_enter_y
                    long_enter_x.append(line[0])
                    long_enter_y.append(np.max((line[1],line[4]))-.1)
                    return_graph_x.append(line[0])
                    return_graph_y.append(returns)
                    local_return_graph_x.append(line[0])
                    local_return_graph_y.append(returns)
                continue

            if long and ((buy_sell < 0.0) or (line[4] < lower_bound)) and not reversal_flag:  #  main buy/sell 2
                #                      on 20x
                usd = (np.min((line[1],line[4])) / position[1]) * position[0]  # assume bad execution
                position = 0
                position = [usd * .9997, np.min((line[1],line[4]))]
                usd = 0.0
                long = False
                in_position = True
                trades += 1
                last_trade_time = iteration
                if line[4] < lower_bound:
                    reversal_flag = True
                    stoploss_triggers += 1
                else:
                    reversal_flag = False
                if plot:
                   #global short_enter_x
                   #global short_enter_y
                    short_enter_x.append(line[0])
                    short_enter_y.append(np.min((line[1],line[4]))+.1)
                    return_graph_x.append(line[0])
                    return_graph_y.append(returns)
                    local_return_graph_x.append(line[0])
                    local_return_graph_y.append(returns)
                continue

        if not desire_position and not in_position:  # section 3
            if not long and ((buy_sell > 0.0) ^ (line[4] > upper_bound)): # moves understanding of long/short desire despite not being in position
                #usd = (position[1] / max(line[1:5])) * position[0]
                #position = 0
                #position = [usd * .9997, max(line[1:5])]
                #usd = 0.0
                long = True
                #in_position = True
                #trades += 1
                last_trade_time = iteration
                if line[4] > ema2[-1] * (1 + variables[2]):
                    reversal_flag = True
                else:
                    reversal_flag = False
                # if plot:
                #    #global long_enter_x
                #    #global long_enter_y
                #     long_enter_x.append(line[0])
                #     long_enter_y.append(line[4]-.1)
                #     return_graph_x.append(line[0])
                #     return_graph_y.append(returns)
                continue

            if long and ((buy_sell < 0.0) ^ (line[4] < lower_bound)):  # moves understanding of long/short desire despite not being in position
                #usd = (min(line[1:5]) / position[1]) * position[0]
                #position = 0
                # position = [usd * .9997, min(line[1:5])]
                # usd = 0.0
                long = False
                # in_position = True
                # trades += 1
                last_trade_time = iteration
                if line[4] < lower_bound:
                    reversal_flag = True
                else:
                    reversal_flag = False
                # if plot:
                #    #global short_enter_x
                #    #global short_enter_y
                #     short_enter_x.append(line[0])
                #     short_enter_y.append(line[4]+.1)
                #     return_graph_x.append(line[0])
                #     return_graph_y.append(returns)
                continue


        if desire_position and not in_position:  # if adx reverses direction, reenter position section 4
            if not long:  #
                #usd = (position[1] / max(line[1:5])) * position[0]
                #position = 0
                position = [usd * .9997, np.min((line[1],line[4]))]
                usd = 0.0
                #long = True
                in_position = True
                trades += 1
                last_trade_time = iteration
                reversal_flag = False
                if plot:
                   #global short_enter_x
                   #global short_enter_y
                    short_enter_x.append(line[0])
                    short_enter_y.append(line[4]+.1)
                    return_graph_x.append(line[0])
                    return_graph_y.append(returns)
                    local_return_graph_x.append(line[0])
                    local_return_graph_y.append(returns)
                continue

            if long:
                #usd = (min(line[1:5]) / position[1]) * position[0]  # assume bad execution
                #position = 0
                position = [usd * .9997, np.min((line[1],line[4]))]
                usd = 0.0
                #long = False
                in_position = True
                trades += 1
                last_trade_time = iteration
                reversal_flag = False
                if plot:
                   #global long_enter_x
                   #global long_enter_y
                    long_enter_x.append(line[0])
                    long_enter_y.append(line[4] - .1)
                    return_graph_x.append(line[0])
                    return_graph_y.append(returns)
                    local_return_graph_x.append(line[0])
                    local_return_graph_y.append(returns)
                continue


        if returns < 50 and not plot:  # fail-quick
            returns = 0
            position = [0, line[4]]
            max_draw_down = -.9271
            break
    ema1_switches = 0
    for index, item in enumerate(ema1):
        if (item > ema1[index-1]) is not (ema1[index-1] > ema1[index-2]):
            ema1_switches += 1
    ema2_switches = 0
    for index, item in enumerate(ema2):
        if (item > ema2[index - 1]) is not (ema2[index - 1] > ema2[index - 2]):
            ema2_switches += 1
    macd_switches = 0
    for index, item in enumerate(buyselllist):
        if (item > 0) is not (buyselllist[index-1] > 0):
            macd_switches += 1
    buyselllist = []
    longest_time_since_last_high = 0
    for index, tick in enumerate(local_return_graph_y):  # TODO TEMPORARY
        if index == 0:
            buyselllist.append(0)
            continue
        #print((tick - local_return_graph_y[index - 1]) / tick)
        buyselllist.append((tick - local_return_graph_y[index - 1]) / tick)
    print(sorted(buyselllist))
    if plot:
        for leverage in range(1, 101):
            max_value = 0
            last_value = 0
            time_of_last_high = local_return_graph_x[0]
            time_since_last_high = 0

            for tickindex, tick in enumerate(local_return_graph_y):
                if tickindex == 0:
                    max_value = tick
                    last_value = tick
                    continue
                delta = (tick - local_return_graph_y[tickindex - 1]) / tick
                delta *= leverage
                last_value *= delta + 1
                if last_value > max_value:
                    time_since_last_high = local_return_graph_x[tickindex] - time_of_last_high
                    time_of_last_high = local_return_graph_x[tickindex]
                    if time_since_last_high > longest_time_since_last_high:
                        longest_time_since_last_high = time_since_last_high
                    max_value = last_value
                if last_value < max_value / float(4):
                    break
            if last_value < max_value / float(4):  # when broken out of the inner for by too much leverage
                max_leverage = leverage - 1
                break
            elif leverage == 100:
                print("got max leverage 100 somehow")
                max_leverage = leverage
                break
    # max_leverage = min(math.floor(-1/min(mdd_relative, -.0001)), max_leverage)



    # print(max(adx), "avg:", np.average(adx), min(adx))
    if plot:
        print (ema1_switches, ema2_switches, "trade signals:", macd_switches, "MDD%", max_draw_down * 100, "dd_rel%:", mdd_relative * 100, "max leverage:", max_leverage)
    else:
        print(ema1_switches, ema2_switches, "trade signals:", macd_switches, "MDD%", max_draw_down * 100, "dd_rel%:",
              mdd_relative * 100)
    #print(max(atr), min(atr))
    #print("min, max, stddev, avg of stddev", min(stddev), max(stddev), np.std(stddev), np.average(stddev))


    if plot:
        global plot_minusdi
        global plot_plusdi
        if train:
            plt.plot(data[int(-len(ema1)-(size/10)):int(-(size/10)), 0], ema1)
            plt.plot(data[int(-len(ema2)-(size/10)):int(-(size/10)), 0], ema2)
            ##global plot_adx

            # plot_adx += adx
            plot_minusdi += minusdi
            plot_plusdi += plusdi
        else:
            plt.plot(data[-len(ema1):, 0], ema1)
            plt.plot(data[-len(ema2):, 0], ema2)
            # global plot_adx

            # plot_adx += adx
            plot_minusdi += minusdi
            plot_plusdi += plusdi
    if long and in_position:
        returns = usd + (np.min((line[1],line[4])) / position[1]) * position[0]
        if plot:
            return_graph_x.append(line[0])
            return_graph_y.append(returns)
            local_return_graph_x.append(line[0])
            local_return_graph_y.append(returns)
    elif in_position:
        returns = usd + (position[1] / np.max((line[1], line[4]))) * position[0]
        if plot:
            return_graph_x.append(line[0])
            return_graph_y.append(returns)
            local_return_graph_x.append(line[0])
            local_return_graph_y.append(returns)
    if not in_position:
        returns = usd
        if plot:
            return_graph_x.append(line[0])
            return_graph_y.append(returns)
            local_return_graph_x.append(line[0])
            local_return_graph_y.append(returns)
    if trades < 20 and not plot and train: returns = 0
    print(list(variables), returns, trades)
    if plot:
        print("Score metric:", returns * (1 + max_draw_down), "stoplosses hit:", stoploss_triggers,
              "longest defecit:", round(longest_time_since_last_high/86400., 2), "days")
    else:
        print("Score metric:", returns * (1 + max_draw_down), "stoplosses hit:", stoploss_triggers)

    return -(returns * (1 + max_draw_down))
# plot_adx = []

def main():
    data = extract_data()
    minutes_train = 1100000  # four zeros = 1 week, five zeroes = 10 weeks,  1100000 = all my data 500000=1year
    #bounds = [(-0, 0), (2, 1000), (-0, 0), (40, 40000), (-1, 1), (10, 40000), (.0001, .002), (10, 40000)]
    #                                                                                atr reverse trigger, lookback 6/7
    bounds = [ (2, 3000), (2, 3000), (0.0, .01), (1, 20), (100, 10000)]
    #          ema1period, ema2period,   SL%,   recent tradeout scale factor, scaling lookback window
    #            0              1         2
    # document for bounds
    fig, ax1 = plt.subplots()
    # result = differential_evolution(objective, bounds, args=(data, minutes_train, True, False), disp=True,
    #                              maxiter=20, popsize=10, callback=callback, strategy='best2bin', polish=False) # todo paralleled popsize was 3
    # print(list(result.x), -result.fun)
    # objective(result.x, data=data, size=minutes_train, train=True, plot=True)
    # test_result = objective(result.x, data=data, size=minutes_train, train=False, plot=True)
    # objective([1299.0, 1358.0, 0.0039968103821166586, 10.507419367673087, 4917.252866397499], data=data, size=minutes_train, train=True, plot=True)
    # test_result = objective([1299.0, 1358.0, 0.0039968103821166586, 10.507419367673087, 4917.252866397499], data=data, size=minutes_train, train=False, plot=True)
    objective([475.0, 1285.0, 0.0018392239365409187, 17.806995940901789, 1670.7339655391988], data=data, size=minutes_train, train=True, plot=True)
    test_result = objective([475.0, 1285.0, 0.0018392239365409187, 17.806995940901789, 1670.7339655391988], data=data, size=minutes_train, train=False, plot=True)
    # print("final test result: ", -test_result)


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
