

import multiprocessing
import os
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

    data = np.array(data)
    # data = data[:-40000] # todo very temp
    # Iterate over the rows, splitting the label from the features.
    # print("iterating over data")
    for index, line in enumerate(data):
        if index > len(data) - (future_target + 1): continue  # ignores 10 minutes at the end
        fvecs.append(line[4])
        if future_target == 1:
            average = data[index+1, 4]
        else:
            average = np.mean(data[index + 1:index + future_target, 4])
        labels.append((average - line[4]) / line[4])
    # print("finalizing load")
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





def mycallback(returntuple):
    distancelist, index = returntuple
    global positions
    positions[index] = (distancelist)
    global callback_calls, pbar
    callback_calls += 1
    pbar.update(1)


def dataproc_worker(index, delta, worker_smoothX):

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
    resultlist = motif_matching_knn.find_matches(worker_smoothX[:-L], query, num_neighbors)
    distancelist = []
    for index2, item in enumerate(resultlist):
        distancelist.append(int(item))


    return (distancelist, index)


def objective(plot=False, do_print=True):
    global adjustment, bnh_returns
    usd = 100.0  # investing 5 dollars at 20x
    position = []
    long = False
    trades = 0
    ema1, ema2, adosc, macd, buyselllist = [], [], [], [], []
    returns = 100
    peak_value=returns
    max_draw_down = 0


    if adjustment is None:
        normalline = mainX[-test_iterations: -test_iterations + 5]
        iterate_lines = 300
        for iterator_local in range(iterate_lines):
            iterator_local -= 100
            compare_data = data[
                           -test_iterations + iterator_local: -test_iterations + iterator_local + 5, 1]
            if np.array_equal(compare_data, normalline):
                adjustment = iterator_local


    ema1_ema, ema2_ema = 0, 0
    global positions_ordered
    if plot:
        global return_graph_x, return_graph_y

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
            sequence = (smoothmainX[prediction + L + 1:prediction + L + max(future_target, 3) + 1]) / smoothmainX[prediction + L] # todo make this take from base not smooth if current run is bad
            average.append(sequence)

        average = np.mean(average, axis=0) #len of future_target *2 -1, current time = index at L
        fit = np.polynomial.polynomial.polyfit(range(len(average)),average, int((future_target/5))+1)
        target = np.polynomial.polynomial.polyval(int(0 + max(future_target/2.0, 1)), fit) \
                 - np.polynomial.polynomial.polyval(0, fit)
        buy_sell = target


        buyselllist.append(buy_sell)
        if ema_timeperiod > 2: # making ema 2 into no ema smoothing
            ema1_ema = (buy_sell - ema1_ema) * (2 / (ema_timeperiod + 1)) + ema1_ema
            buy_sell = ema1_ema
        # end programmable bits, begin simulation
        if index == 0 or position is None or position == []:
            bnh_entry = [usd * .9997, line]
            if buy_sell > 0:  # only executed on first run, from dollars to btc/shorts
                position = [usd * .9997, line]  # store entry price
                usd = 0
                long = True
                if plot:
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
                    short_enter_x.append(index)
                    short_enter_y.append(line + .1)
                    return_graph_x.append(index)
                    return_graph_y.append(100)
                continue
        if position is None:
            print(index, position, line, adjustment, usd, returns, long)
            raise Exception("position was somehow not len 2")
        if not long and buy_sell > 0:
            usd = (position[1] / max_exec_price) * position[0]
            position = [usd * .9997, max_exec_price]
            usd = 0.0
            long = True
            trades += 1
            if plot:
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

    macd_switches = 0
    for index2, item in enumerate(buyselllist):
        #print(item) #
        if (item > 0) is not (buyselllist[index2 - 1] > 0):
            macd_switches += 1
    if plot:

        print("mdd:", max_draw_down)
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
    f1 = sns.jointplot(to_plot[:, 1], to_plot[:, 0], color="r", kind="kde").plot_joint(
        sns.regplot, marker="+", scatter_kws={'alpha': 1.})
    plt.title("Predicted vs. Actual Price Deltas")
    f1.set_axis_labels("Actual", "Predicted")
    print("correct percentage, W, l", correct_predictions / test_iterations, W, L)
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
    ax2.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())


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
        score = -objective(do_print=False)
        if score > oldscore:
            objective_scores = [score, ema_timeperiod]
            oldscore = score

        ema_timeperiod = round(ema_timeperiod * 1.3)

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
    objective_scores = []
    oldscore = 0
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
    smoothX = DWT_smooth(X)
    smoothmainX = DWT_smooth(mainX)


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

if __name__ == '__main__':

    run_stuff()
