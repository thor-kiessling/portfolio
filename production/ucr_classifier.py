

import multiprocessing
import os
from subprocess import run, PIPE, DEVNULL
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import collections
import pywt
import datetime
#import ucrdtw
from scipy.optimize import differential_evolution
import motif_matching_knn
import time
import pickle
from tqdm import tqdm
from array import array

import globalvar

mode = "verify single"   # valid options "verify single" "verify winners" "optimize"
print("mode = " + mode)
os.nice(39)

long_enter_x = []
short_enter_x = []
long_enter_y = []
short_enter_y = []


def extract_data():
    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []
    # print("loading from DB")
    globalvar.dbcur.execute("SELECT * FROM bmxswap ORDER BY time ASC")
    data = globalvar.dbcur.fetchall()

    # dt = np.dtype(np.double)
    # print("converting from DB")
    data = np.array(data)
    # data = data[:-40000] # todo very temp
    # Iterate over the rows, splitting the label from the features.
    # print("iterating over data")
    for index, line in enumerate(data):
        if index > len(data) - (future_target + 1): continue  # ignores 10 minutes at the end
        # print(len(temp_list))
        # print(len(fvecs))
        fvecs.append(line[4])
        if future_target == 1:
            average = data[index+1, 4]
        else:
            average = np.mean(data[index + 1:index + future_target, 4])
        labels.append((average - line[4]) / line[4])
        # print(fvecs)
        # if index % 1000 == 0:
        #     print(index / len(data))
    # print("finalizing load")
    # print(data.shape)
    # print(np.array(fvecs).shape)
    if True in np.isnan(labels):
        raise ValueError("nan in the labels")
    if True in np.isnan(fvecs):
        raise ValueError("nan in the fvecs")
    return np.array(fvecs), np.array(labels), data


test_iterations = 500  # false positives up to 2k proven. 40k eval
ema_timeperiod = 2  # ema timeperiod initial

W = .106574844973  # Warp factor, %, between 1 and 30?, HEAVILY affects runtime. .05 = 2.5s, .15 = 7.5s, .3 = incomplete after a minute
L = 110  # sequence length, effects runtime
future_target = 15  # targets average of n+1 to n+future_target
decomp_level = 3
wavelet = None
wavelet_pad_mode = "symmetric"
wavelet_initial = 2
num_neighbors = 3  # how many neighbors to get from DTW, aka n/final arg
# returns, correct percentage, bnh returns, W, l, target, decomp, wavelet pad, wavelet, initial, knn, ema_time 118.47 0.5041 83.6 0.106574844973 144 39 3 symmetric None 2 3 22
#  '158.53', 0.5043, '158.53', 0.11664191152845935, 440, 38, 4, 'constant', None, 1, 10, 22)
# '158.42', 0.51715, '158.53', 0.11881114609914123, 59, 38, 6, 'symmetric', None, 1, 4, 13) garbage
# '157.88', 0.5108, '157.99', 0.10793797041541886, 451, 56, 6, 'constant', None, 1, 5, 13)
# , '158.53', 0.50645, '158.53', 0.016812383040889917, 441, 38, 4, 'symmetric', None, 2, 11, 29) garbage
#  '118.15', 0.50865, '116.63', 0.048356169086347239, 365, 40, 4, 'periodic', None, 2, 14, 29) garbage 40k ret
# 40k winner ^

#  '116.38', 0.501125, '116.38', 0.084118591365885551, 432, 50, 3, 'reflect', None, 2, 8, 22)

# 140.42 0.50782 88.94 0.152662767305 321 24 4 periodic rbio2.2 1 3 64 super serial winner fo real # seems majority short trying longer


# , '126.88', 0.682, '109.79', 0.18445573256877923, 52, 55, 5, 'periodic', 'db6', 1, 6, 49) acc .533 ret 200 dd 45% pr .025 seems to be
# returns, correct percentage, bnh returns, W, l, target, decomp, wavelet pad, wavelet, initial, knn, ema_time 103.67 0.50332 87.95 0.00336027101107 259 58 6 reflect None 2 3 5
# returns, correct percentage, bnh returns, W, l, target, decomp, wavelet pad, wavelet, initial, knn, ema_time 108.49 0.50062 88.46 0.0821515448902 279 7 6 periodic rbio2.2 1 3 64
# returns, correct percentage, bnh returns, W, l, target, decomp, wavelet pad, wavelet, initial, knn, ema_time 108.04 0.50258 88.37 0.0374794426246 189 13 5 periodic rbio2.2 2 2 64
#


# , '105.22', 0.5126, '95.72', 0.072740030455287041, 298, 17, 6, 'symmetric', 'sym3', 1, 9, 400)
# 10k: ret 124 ema 1932 19tx pr .039
# (,'103.77', 0.5084, '96.42', 0.10390992823201763, 216, 7, 5, 'reflect', 'sym3', 1, 3, 520)
#
# , '104.34', 0.532, '95.72', 0.072740030455287041, 298, 17, 4, 'symmetric', 'sym3', 1, 6, 182)
# 10k: ret 108 ema 1932 32tx pr .039
# , '104.23', 0.5124, '95.72', 0.091029962369990558, 296, 17, 3, 'smooth', 'bior2.2', 2, 10, 879)
# bad ret neg pr
#, '105.64', 0.552, '96.95', 0.1275921179087795, 558, 38, 3, 'symmetric', None, 1, 2, 400)
#
# , '106.23', 0.554, '96.61', 0.082850588862355204, 360, 46, 4, 'smooth', 'bior2.2', 1, 2, 49)
# , '121.22', 0.507575, '213.39', 0.053165347377514972, 341, 30, 4, 'symmetric', 'bior2.2', 1, 8, 520)
# 10k: ret 108 ema 237 pr .09 tx 61 40k: ret 113 ema 520 acc .506
# , '187.66', 0.512525, '213.27', 0.0085584461532686729, 569, 34, 2, 'smooth', 'db9', 1, 13, 879)
# 40k: ret 213 ema 1143 acc .5134 pr.007   BEATS BUY AND HOLD!!!!!!! mdd24% not better than bm15


# only W can be varied in the program right now

# possible train idea: decrease W if correct distances > wrong distances, increase if opposite
# .1W and 1000L gives Distance ~ 2.12, .2 gives 1.85, .01 gives ~4.2
# all of these are results from 10k





def mycallback(returntuple):
    distancelist, index = returntuple
    global positions
    # if distance == 1337:
    #     # print("error in execution at", position, distance, callback_calls)
    #     print(position)
    #     return  # don't add it to positions
    positions[index] = (distancelist)
    global callback_calls, pbar
    callback_calls += 1
    pbar.update(1)
    # print("absolute distance between index and query " + str(index) + ": "
    #       + str(abs(mainX[len(mainX) - len(testX) + index - L]
    #                 - mainX[position])))  # not needed I am satisfied with DTW being scale-independent

        # print(callback_calls / len(testY))


def dataproc_worker(index, delta, worker_smoothX):
    # start = time.time()

    to_smooth = mainX[0: len(mainX) - len(testX) + index]
    smoothed = DWT_smooth(to_smooth)
    difference = len(smoothed) - len(to_smooth)
    least_distance = float('inf')
    best_offset = 0
    for iteration in range(difference * 2):
        if iteration == 0:
            smooth_to_compare = smoothed[-L:]
        else:
            smooth_to_compare = smoothed[-(L + iteration):-iteration]
        comparison = to_smooth[-L:]
        distance = np.linalg.norm(smooth_to_compare - comparison)
        if distance < least_distance:
            best_offset = iteration
            least_distance = distance
    if best_offset == 0:
        query = smoothed[-L:]
    else:
        query = smoothed[-(L + best_offset):-best_offset]
    # end = time.time()
    # print(str(index) + " smoothing and searching took " + str(end - start))
    # comparison = smoothmainX[len(mainX) - len(testX) + index - L: len(mainX) - len(testX) + index]
    # plt.plot(comparison, "r-")
    # plt.plot(query, "k-")
    # plt.show()

    # np.savetxt("Query" + str(index) + ".txt", query, delimiter='\t')
    # runresult = run(["nice", "-15", "./UCR_DTW_KNN.so", "Data" + str(L) + ".txt", "Query" + str(index) + ".txt", str(L), str(W), str(num_neighbors)],
    #                 stderr=DEVNULL, stdout=PIPE,
    #                 universal_newlines=True)
    # os.remove("Query" + str(index) + ".txt")
    # start = time.time()
    resultlist = motif_matching_knn.find_matches(worker_smoothX[:-L], query, num_neighbors)
    # end = time.time()
    # print(str(index) + " find_matches took " + str(end - start))

    # print(runresult)
    # position, distance = _ucrdtw.ucrdtw(smoothX, smoothmainX[len(mainX) - len(testX) + index - L: len(mainX) - len(testX) + index], L, W)
    # resultlines = runresult.stdout
    # resultlines = resultlines.split('\n')
    distancelist = []
    # distancelist = resultlines[3:3+num_neighbors]
    # for result in resultlines:
    #     split = result.split(":")
    #     if split[0] == 'Location ':
    #         position = int(split[1])
    #     if split[0] == 'Distance ':
    #         distance = float(split[1])
    for index2, item in enumerate(resultlist):
        distancelist.append(int(item))
    # if distance is None or (distance == 0 and position == 0):
    #     distance = 1337
    #     return (str(resultlines), distance, index)
    # if position == 0 or position is None:
    #     position = 1337


    return (distancelist, index)


def objective(plot=False, do_print=True):
    # variables[1], variables[3], variables[6] = int(variables[1]), int(variables[3]), int(variables[6])
    global adjustment, bnh_returns
    usd = 100.0  # investing 5 dollars at 20x
    position = []
    long = False
    trades = 0
    ema1, ema2, adosc, macd, buyselllist = [], [], [], [], []
    returns = 100
    peak_value=returns
    max_draw_down = 0

    # globalvar.dbcur.execute("SELECT * FROM bmxswap ORDER BY time ASC")
    # data = globalvar.dbcur.fetchall()
    # data = np.array(data)

    if adjustment is None:
        normalline = mainX[-test_iterations: -test_iterations + 5]
        iterate_lines = 300
        for iterator_local in range(iterate_lines):
            iterator_local -= 100
            compare_data = data[
                           -test_iterations + iterator_local: -test_iterations + iterator_local + 5, 1]
            if np.array_equal(compare_data, normalline):
                adjustment = iterator_local


    # if abs(variables[0]) > abs(variables[2]) + abs(variables[4]): return 0
    # variables[0], variables[2], variables[4] = 0,0,0\
    ema1_ema, ema2_ema = 0, 0
    # gnd(distance)  ## for reference v
    global positions_ordered
    if plot:
        global return_graph_x, return_graph_y

        # for index, line in enumerate(data):
        #     if index > len(data) - (future_target + 1): continue  # ignores 10 minutes at the end
        # data = data[:- (future_target + 1)]
    for index, position_distance in positions_ordered.items():
        predictionlist = position_distance
        line = data[-test_iterations + index + adjustment, 1]
        if plot:
            rawline = data[-test_iterations + index + adjustment]
        prevline = data[-test_iterations + index + adjustment - 1, 1]
        nextline = data[-test_iterations + index + adjustment + 1, 1]
        if future_target == 1 and index == len(positions_ordered) - 1:
            nextline2 = nextline
        else:
            nextline2 = data[-test_iterations + index + adjustment + 2, 1]
        min_exec_price, max_exec_price = min(line, nextline, nextline2, prevline), max(line, nextline, nextline2, prevline)
        average = []
        # average2 = []
        for prediction in predictionlist:
            # if future_target == 1:
            #     average.append((smoothmainX[prediction + L + 1] - smoothmainX[prediction + L]) / smoothmainX[prediction + L])
            # else:
            # average.append(
            #     (np.mean(smoothmainX[prediction + L + 1:prediction + L + future_target]) - smoothmainX[prediction + L])/
            #     smoothmainX[prediction + L])
            #subseq1 = data[-test_iterations + index + adjustment :-test_iterations + index + adjustment + future_target, 1]/data[-test_iterations + index + adjustment, 1]
            # subseq1 = data[prediction + L + 1:prediction + L + max(future_target, 3) + 1, 1]/data[prediction + L, 1]
            sequence = (smoothmainX[prediction + L + 1:prediction + L + max(future_target, 3) + 1]) / smoothmainX[prediction + L] # todo make this take from base not smooth if current run is bad
            # sequence = np.concatenate((subseq1,subseq2))
            # average2.append(subseq1)
            average.append(sequence)

        average = np.mean(average, axis=0) #len of future_target *2 -1, current time = index at L
        # average2 = np.mean(average2, axis=0)
        fit = np.polynomial.polynomial.polyfit(range(len(average)),average, int((future_target/5))+1)
        target = np.polynomial.polynomial.polyval(int(0 + max(future_target/2.0, 1)), fit) \
                 - np.polynomial.polynomial.polyval(0, fit)
        buy_sell = target

        # plt.plot(range(len(average)), average, "b-")
        # xlist = range(len(average))
        # plt.axvline(x=0)
        # plt.axvline(x=int(future_target/2))
        # plt.plot(range(len(average)), np.polynomial.polynomial.polyval([xlist], fit)[0,:], "r-" )
        # # plt.plot(range(len(average)), average2, "m-")
        # plt.plot(range(len(average)), data[-test_iterations + index + adjustment:
        # -test_iterations + index + adjustment +future_target , 1]/data[-test_iterations + index + adjustment, 1], "g-")
        # plt.show()

        buyselllist.append(buy_sell)
        if ema_timeperiod > 2: # making ema 2 into no ema smoothing
            ema1_ema = (buy_sell - ema1_ema) * (2 / (ema_timeperiod + 1)) + ema1_ema
            buy_sell = ema1_ema
        # print(buy_sell)
        # end programmable bits, begin simulation
        if index == 0 or position is None or position == []:
            bnh_entry = [usd * .9997, line]
            if buy_sell > 0:  # only executed on first run, from dollars to btc/shorts
                position = [usd * .9997, line]  # store entry price
                usd = 0
                long = True
                if plot:
                    # global long_enter_x
                    # global long_enter_y
                    long_enter_x.append(index)
                    long_enter_y.append(line - .1)
                    return_graph_x.append(index)
                    return_graph_y.append(100)
                continue
            else:
                position = [usd * .9997, line]  # store entry price
                usd = 0
                long = False
                if plot:
                    # global short_enter_x
                    # global short_enter_y
                    short_enter_x.append(index)
                    short_enter_y.append(line + .1)
                    return_graph_x.append(index)
                    return_graph_y.append(100)
                continue
        # if len(position) == 2:
        #     if long:
        #         returns = usd + (line / position[1]) * position[0]
        #     if not long:
        #         returns = usd + (position[1] / line) * position[0]
        if position is None:
            print(index, position, line, adjustment, usd, returns, long)
            raise Exception("position was somehow not len 2")
        if not long and buy_sell > 0:
            usd = (position[1] / max_exec_price) * position[0]
            # position = 0
            position = [usd * .9997, max_exec_price]
            usd = 0.0
            long = True
            trades += 1
            if plot:
                # global long_enter_x
                # global long_enter_y
                long_enter_x.append(index)
                long_enter_y.append(max_exec_price - .1)
                return_graph_x.append(index)
                return_graph_y.append(returns)
                print(datetime.datetime.utcfromtimestamp(rawline[0]), max_exec_price, "long")
        if long and buy_sell < 0:  # simple buy and sell, assume bad executions, and .03% fee on full contract value, .6%
            #                      on 20x
            usd = (min_exec_price / position[1]) * position[0]  # assume bad execution
            # position = 0
            position = [usd * .9997, min_exec_price]
            usd = 0.0
            long = False
            trades += 1
            if plot:
                # global short_enter_x
                # global short_enter_y
                short_enter_x.append(index)
                short_enter_y.append(min_exec_price + .1)
                return_graph_x.append(index)
                return_graph_y.append(returns)
                print( datetime.datetime.utcfromtimestamp(rawline[0]), min_exec_price, "short")
        if len(position) is 2:
            if long:
                returns = usd + (min_exec_price / position[1]) * position[0]
            else:
                returns = usd + (position[1] / max_exec_price) * position[0]
            if returns > peak_value:
                peak_value = returns
            draw_down = (returns - peak_value) / peak_value
            if draw_down < max_draw_down:
                max_draw_down = draw_down
                # if returns < 50 and not plot:  # fail-quick
                #     returns = 0
                #     position[0] = 0
                #     break

    macd_switches = 0
    for index2, item in enumerate(buyselllist):
        #print(item) #
        if (item > 0) is not (buyselllist[index2 - 1] > 0):
            macd_switches += 1
    if plot:

        print("mdd:", max_draw_down)
    # if do_print:
    #     print("macd switches", macd_switches)

    # plt.plot(data[-len(ema1):, 0], ema1)
    # plt.plot(data[-len(ema2):, 0], ema2)
    if len(position) is 2:
        bnh_returns = (min_exec_price / bnh_entry[1]) * bnh_entry[0]
        if long:
            returns = usd + (min_exec_price / position[1]) * position[0]
            if plot:
                return_graph_x.append(index)
                return_graph_y.append(returns)
        else:
            returns = usd + (position[1] / max_exec_price) * position[0]
            if plot:
                return_graph_x.append(index)
                return_graph_y.append(returns)
    if do_print:
        print(returns, trades, "/", macd_switches)
        print(max(buyselllist), min(buyselllist))
    buyselllist = []

    if trades < 5:
        returns = 0
    return returns * -1


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


acc_W = []
positions = {}
distances_correct = []
distances_wrong = []
iterator = []
correct_predictions = 0
callback_calls = 0
positions_ordered = {}
smoothmainX, smoothX, smoothtestX = [], [], []
X, Y, testX, testY = [], [], [], []
mainX, mainY = [], []
return_graph_x, return_graph_y = [], []
to_plot = []
adjustment = None
bnh_returns = 100

def main():
    global mainX, mainY, X, Y, testX, testY, to_plot
    global L, W, future_target, decomp_level, wavelet, wavelet_pad_mode, wavelet_initial

    shared_startup(True)
    to_plot = np.array(to_plot)
    # f1 = sns.regplot(, fit_reg=True)  # x = actual, y = predicted
    #f1 = plt.figure(1)
    f1 = sns.jointplot(to_plot[:, 1], to_plot[:, 0], color="r", kind="kde").plot_joint(
        sns.regplot, marker="+", scatter_kws={'alpha': 1.})
    plt.title("Predicted vs. Actual Price Deltas")
    f1.set_axis_labels("Actual", "Predicted")
    print("correct percentage, W, l", correct_predictions / test_iterations, W, L)
    # print("wrong distances:", np.mean(distances_wrong))
    # print("correct distances:", np.mean(distances_correct))
    objective_scores = []
    oldscore = 0
    global ema_timeperiod
    # if ema_timeperiod == 2:
    while ema_timeperiod < 2000:
        print("trying with ema", ema_timeperiod)
        score = -objective()
        if score > oldscore:
            objective_scores = [score, ema_timeperiod]
            oldscore = score

        ema_timeperiod = round(ema_timeperiod * 1.3)

    ema_timeperiod = objective_scores[1]
    print("using winning ema period:", ema_timeperiod)
    objective(plot=True)
    print("buy and hold returns: ", bnh_returns)
    # acc_W.append([correct_predictions/test_iterations, W])
    # plt.plot(acc_W[:, 1], acc_W[:, 0], label="fut"+str(future_target)+"L"+str(L)+"sv.txt")
    # np.savetxt("fut"+str(future_target)+"L"+str(L)+"sv.txt", acc_W)

    # directory = os.listdir()
    # saves = fnmatch.filter(directory, 'fut*sv.txt')
    # for save in saves:
    #     loaded = np.loadtxt(save)
    #     plt.plot(loaded[:, 1], loaded[:, 0], label=save)
    plt.figure(1)
    fig, ax1 = plt.subplots()
    plt.title("Step-forward Simulated Performance")
    plt.ylabel("Bitcoin Price")
    plt.xlabel("Time (minutes)")
    x = range(test_iterations)
    if adjustment != 0:
        y = data[adjustment - test_iterations:adjustment, 1]
    else:
        y = data[-test_iterations:, 1]
    plt.plot(x, y, 'k-')

    plt.plot(long_enter_x, long_enter_y, 'g^')
    plt.plot(short_enter_x, short_enter_y, 'rv')
    ax2 = ax1.twinx()
    ax2.plot(return_graph_x, return_graph_y, color='b')
    ax2.set_ylabel("Cumulative Returns", color="b")
    ax2.set_yscale('log')
    ax2.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=range(10)))
    ax2.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=range(100)))
    ax2.tick_params(axis='y', colors='b', which='both')
    # ax2.yaxis.set_majorticklabels([])
    ax2.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    # ax2.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatter())


    plt.show()

candidates = []
accuracy_returns = [[.5, 100]]
def optimize(variables):
    # print("trying", variables)
    global mainX, mainY, X, Y, testX, testY, num_neighbors, accuracy_returns
    global L, W, future_target, decomp_level, wavelet, wavelet_pad_mode, wavelet_initial



    wavelets = [None, "db3", 'db6', 'db9', "bior2.2", "coif3", "dmey", "rbio2.2", "sym3"] # removed haar too many divide by 0
    wavelet_pad_modes = ['constant', 'symmetric', 'reflect', "periodic", "smooth"]
    L = int(variables[0])
    W = variables[1]
    future_target = int(variables[2])
    decomp_level = int(variables[3])
    wavelet = wavelets[int(variables[4])]
    wavelet_pad_mode = wavelet_pad_modes[int(variables[5])]
    wavelet_initial = min(int(variables[6]), int(variables[3]-1))
    num_neighbors = int(variables[7])

    shared_startup(False)
    global ema_timeperiod, correct_predictions, positions_ordered
    ema_timeperiod = 2
    objective_scores = [0, 0]
    oldscore = 0
    while ema_timeperiod < 65:
        # print("trying with ema", ema_timeperiod)
        score = -objective(do_print=False)
        if score > oldscore:
            objective_scores = [score, ema_timeperiod]
            oldscore = score

        ema_timeperiod = round(ema_timeperiod * 1.3)
    # os.remove("Data" + str(L) + ".txt")

    best_ema_timeperiod = objective_scores[1]
    best_returns = objective_scores[0]
    accuracy_returns = np.array(accuracy_returns)

    accuracy_returns = np.append(accuracy_returns, [[correct_predictions / test_iterations, best_returns]], axis=0)
    if len(accuracy_returns) != 1:
        corr = np.corrcoef(accuracy_returns[:, 0], accuracy_returns[:, 1])
        corr = corr[1, 0]
    else:
        corr = 0
    global candidates, adjustment
    if  best_returns-100 > abs(bnh_returns-100): # correct_predictions/test_iterations > .5 and
        candidates.append(variables)
        print("The line below this one is an accepted candidate for further processing")
    print("returns, correct percentage, bnh returns, W, l, target, decomp, wavelet pad, wavelet, initial, knn, ema_time",
          str(round(best_returns, 2)), correct_predictions / test_iterations, str(round(bnh_returns, 2)), W, L,
          future_target, decomp_level, wavelet_pad_mode, wavelet, wavelet_initial, num_neighbors, best_ema_timeperiod)
    # print("accuracy-returns correlation " + str(corr))
    # print("wrong distances:", np.mean(distances_wrong))
    # print("correct distances:", np.mean(distances_correct))
    objective_scores = []
    oldscore = 0
    # return_thing = -correct_predictions/test_iterations
    return_thing = -best_returns
    correct_predictions = 0
    positions_ordered = {}
    iterator = []
    adjustment = None
    return return_thing
pbar = None

def shared_startup(ismain=False):
    global mainX, mainY, X, Y, testX, testY, data
    mainX, mainY, data = extract_data()

    X, Y, testX, testY = [], [], [], []
    length = len(mainX)
    # X, testX, Y, testY = sklearn.model_selection.train_test_split(mainX, mainY, test_size=0.2, random_state=42)
    for index, line in enumerate(mainX):
        if index < length - test_iterations :
            X.append(line)
        elif index > length - test_iterations:
            testX.append(line)

    for index, line in enumerate(mainY):
        if index < length - test_iterations :
            Y.append(line)
        elif index > length - test_iterations:
            testY.append(line)
    X = np.array(X)
    Y = np.array(Y)
    testX = np.array(testX)
    testY = np.array(testY)

    global smoothmainX, smoothX, smoothtestX
    # smoothtestX = DWT_smooth(testX)
    smoothX = DWT_smooth(X)
    smoothmainX = DWT_smooth(mainX)
    # while len(testX) < len(smoothtestX):
    #     smoothtestX = np.delete(smoothtestX, -1)
    #
    # while len(X) < len(smoothX):
    #     smoothX = np.delete(smoothX, -1)
    #
    # while len(mainX) < len(smoothmainX):
    #     smoothmainX = np.delete(smoothmainX, -1)

    # smoothX, smoothmainX, smoothtestX = np.array(smoothX), np.array(smoothmainX), np.array(smoothtestX)
    # smoothmainX = np.array(smoothmainX)
    # np.savetxt("Data" + str(L) + ".txt", smoothX, delimiter='\t')
    # print(X.shape, Y.shape, mainX.shape, mainY.shape, testX.shape, testY.shape)
    # print(smoothX.shape, smoothmainX.shape, smoothtestX.shape)

    # plt.plot(testX, "r-")
    # plt.plot(smoothtestX, "k-")
    # plt.title(decomp_level)
    # plt.show()

    # if run displays the end item, then it failed to find a match
    # plot Accuracy vs W

    # test = motif_matching_knn.zNormalize(np.array(testX))
    # print(np.mean(testX), np.std(testX), len(testX))
    # print(np.mean(test), np.std(test), len(test))
    # plt.plot(range(399), testX)
    # plt.figure(2)
    # plt.plot(range(399), test)
    # plt.show()


    global pbar
    pbar = tqdm(total=test_iterations-1, desc=str(L), leave=True)

    global correct_predictions, to_plot, callback_calls
    if not ismain:
        global iterator
        iterator = []
    for index, delta in enumerate(testY):
        iterator.append((index, delta, smoothX))
    pool = multiprocessing.Pool(processes=12)
    for argument in iterator:
        pool.apply_async(dataproc_worker, argument, callback=mycallback)
    pool.close()
    pool.join()
    callback_calls = 0
    pbar.close()

    to_plot = []
    global positions_ordered, positions
    positions_ordered = collections.OrderedDict(sorted(positions.items(), key=lambda t: t[0]))
    positions = {}
    for index, position_distance in positions_ordered.items():
        positionlist = position_distance
        average = []
        for position in positionlist:
            if future_target == 1:
                average.append((smoothmainX[position + L + 1] - smoothmainX[position + L]) / smoothmainX[position + L])
            else:
                average.append(
                    (np.mean(smoothmainX[position + L + 1:position + L + future_target]) - smoothmainX[position + L]) /
                    smoothmainX[position + L])
        average = np.mean(average)
        try:
            if average == 0 and testY[index] == 0:
                correct_predictions -= .5
            if (average > 0) == (testY[index] > 0):
                correct_predictions += 1
            if ismain:
                to_plot.append([average, testY[index]])
        except IndexError:
            continue



def callback(xk, convergence=None):
    print("generation best:", list(xk), convergence)
    return


def run_stuff():
    if mode == "verify single":
        main()
        return
    if mode == "verify winners":
        global test_iterations, candidates
        for files in os.listdir("."):
            if files.startswith("winner"):
                print(files)
                candidates.append(pickle.load(open(files, 'rb')))
        candidates_local = candidates
        candidates = []
        test_iterations = 30000
        print("now running 10k: ", str(len(candidates_local)))
        for candidate in candidates_local:
            optimize(candidate)
        candidates_local = candidates
        candidates = []
        test_iterations = 40000
        print("now running 40k: ", str(len(candidates_local)))
        for candidate in candidates_local:
            optimize(candidate)
        candidates_local = candidates
        candidates = []
        test_iterations = 50000
        print("now running 50k: ", str(len(candidates_local)))
        for candidate in candidates_local:
            optimize(candidate)
        if len(candidates) > 3:
            candidates_local = candidates
            candidates = []
            test_iterations = 60000
            print("now running 60k: ", str(len(candidates_local)))
            for candidate in candidates_local:
                optimize(candidate)
        return
    # L = 30  # sequence length, effects runtime
    # W = .15  # Warp factor, %, between 1 and 30?, HEAVILY affects runtime. .05 = 2.5s, .15 = 7.5s, .3 = incomplete after a minute
    # future_target = 10  # targets average of n+1 to n+future_target
    # ema_timeperiod = 2  # ema timeperiod initial
    # decomp_level = 4
    if mode == "optimize":
        while True:

            #           L          W          future_target     decomp_level   wavelet    wavelet pad mode wavelet initial num_neighbors
            test_iterations = 40000
            candidates = []
            bounds = [(10, 500), (.001, .20),   (1, 60),         (2, 7),         (0, 8.9),     (0, 4.9),     (1, 3), (1, 15)]
            result = differential_evolution(optimize, bounds, args=(), disp=True,
                                            maxiter=20, popsize=3, callback=callback, strategy='best2bin', polish=False)

            # candidates_local = candidates[len(candidates)/2:]
            # candidates = []
            # test_iterations = 5000
            # print("now running 5k: ", str(len(candidates_local)))
            # for candidate in candidates_local:
            #     optimize(candidate)
            candidates_local = candidates
            candidates = []
            test_iterations = 50000
            print("now running 50k: ", str(len(candidates_local)))
            for candidate in candidates_local:
                optimize(candidate)

            if candidates:
                print("GOT WINNER WOOOOOO")
                for candidate in candidates:
                    pickle.dump(candidate, open('winner'+str(candidate[0]), 'wb'))
            else:
                print("no winners")

        # optimize(result.x, )
        # main()
if __name__ == '__main__':

    run_stuff()
