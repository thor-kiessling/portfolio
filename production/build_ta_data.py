import csv
import multiprocessing
from datetime import datetime, timedelta
from multiprocessing import Manager
from random import randrange

import numpy as np
import talib.stream as ta

import get_history
import globalvar
import os
os.nice(15)


# os.system("taskset -p 0xff %d" % os.getpid())
# os.nice(19)


# def mytrace(cursor, statement, bindings):
#    "Called just before executing each statement"
#    # print("SQL:", statement)
#    if bindings:
#        # print("Bindings:", bindings)
#    return True  # if you return False then execution is aborted


def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    if end > start:
        delta = end - start
    else:
        delta = start - end
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)


def second_range(start=None, end=None):
    span = end - start
    span = int(span.total_seconds())
    for i in range(span + 1):
        yield start + timedelta(seconds=i)


def date_range(start=None, end=None):  # hours from get_hist
    span = end - start
    for i in range(int(span.seconds / 3600) + int(span.days * 24) + 2):
        yield start + timedelta(seconds=3600 * i)


def genrandomtime(data_start, now, offset):
    while True:
        end = random_date(data_start, now)
        start = end - offset
        if start < data_start: continue
        return (start, end)


def flatten(input_, output=None):
    if output is None:
        output = []
    if isinstance(input_, str):
        output.append(input_)
    else:
        for item in input_:
            try:
                flatten(item, output)
            except TypeError:
                output.append(item)
    return output


def dataproc_worker(start, end, live, ns, data=None):
    while True:
        # print(start,end, "gonna call the DB and get:")
        start = start.replace(microsecond=0)
        end = end.replace(microsecond=0)
        flip = False
        # if not live:
        #     if randrange(0, 10) > 4: flip = True
        #     else: flip = False
        if data is None:
            globalvar.dbcur.execute("SELECT * FROM futures WHERE time BETWEEN (?) AND (?)",
                                    [(start - datetime(1970, 1, 1)) / timedelta(seconds=1),
                                     (end - datetime(1970, 1, 1)) / timedelta(seconds=1)])
            data = globalvar.dbcur.fetchall()


        # benchstart = datetime.utcnow()
        data = [list(elem) for elem in data]
        # print("#1:", data[0], data[-1])
        # for index, line in enumerate(data):
        #     # data[index][0] = datetime.strptime(line[0], "%Y-%m-%dT%H:%M:%SZ") #problem 2
        #     date = line[0]  # string of '2016-07-07T10:36:42Z' using explicit slicing:
        #     data[index][0] = datetime(int(date[:4]), int(date[5:7]), int(date[8:10]), int(date[11:13]),
        #                               int(date[14:16]), int(date[17:19]))
        #

        averageNumerator = 0.0
        # print("#2:",data[0],data[-1])
        datalength = len(data)
        if not live:
            finaltime = len(data) - 120
        # print(data[-1][0], "as finaltime", data[-1][1], "as lastlast") # consistently accurate
        for index, line in enumerate(data):
            if live:
                averageNumerator += line[4]
            elif index < finaltime:
                averageNumerator += line[4]
        # if (data[-1][0] == data[-120][0]): break
        # here is where we do live adjustments remove 6k seconds at start pad 6k 0's at end
        # print("#3", data[0], data[-1], len(data))
        averagePrice = averageNumerator / (len(data) - 120)
        if live:
            del data[0:121]

        for index, line in enumerate(data):  # normalize to the 120 mins average price
            data[index][1] = line[1] / averagePrice  # o
            data[index][2] = line[2] / averagePrice  # h
            data[index][3] = line[3] / averagePrice  # l
            data[index][4] = line[4] / averagePrice  # c
        timeperiods = [10, 100, 1000, 10000,
                       100000]  # (len(data) - 6000)] is too long, 100k max in C talib
        # initlist = [data[0][0]] * 86400
        # # print (timeperiods)
        if not live:
            open = np.array([row[1] for row in data[0:-120]], dtype=np.double)
            high = np.array([row[2] for row in data[0:-120]], dtype=np.double)
            low = np.array([row[3] for row in data[0:-120]], dtype=np.double)
            close = np.array([row[4] for row in data[0:-120]], dtype=np.double)
            volume = np.array([row[5] for row in data[0:-120]], dtype=np.double)
        else:
            open = np.array([row[1] for row in data[0:-1]], dtype=np.double)
            high = np.array([row[2] for row in data[0:-1]], dtype=np.double)
            low = np.array([row[3] for row in data[0:-1]], dtype=np.double)
            close = np.array([row[4] for row in data[0:-1]], dtype=np.double)
            volume = np.array([row[5] for row in data[0:-1]], dtype=np.double)
            # print("internal last:", close[-1])
        if flip:
            open = (open * -1) + 2
            high = (high * -1) + 2
            low = (low * -1) + 2
            close = (close * -1) + 2
            data = np.array([row[4] for row in data[0:-1]], dtype=np.double)
            data = (data * -1) +2
        # let's do the non-timeperiod indicators first
        # print("problem four took " + str(datetime.utcnow() - benchstart))
        # print("starting TA")
        trange = ta.TRANGE(high, low, close)
        obv = ta.OBV(close, volume)
        ad = ta.AD(high, low, close, volume)
        ht_trendline = ta.HT_TRENDLINE(close)
        ht_dcperiod = ta.HT_DCPERIOD(close)
        ht_dcphase = ta.HT_DCPHASE(close)
        ht_phasor_inphase, ht_phasor_quadrature = ta.HT_PHASOR(close)
        ht_sine, ht_leadsine = ta.HT_SINE(close)
        ht_trendmode = ta.HT_TRENDMODE(close)
        mama, fama = ta.MAMA(close, fastlimit=.5, slowlimit=.05)
        sar = ta.SAR(high, low)  # could crash, no extended params supplied
        sarext = ta.SAREXT(high, low)  # this one too
        bop = ta.BOP(open, high, low, close)
        # print("basic TA took " + str(datetime.utcnow() - benchstart))
        # print("starting timeperiod TA")

        # setting up timeperiod-requiring indicators
        slowk, slowd, fastk, fastd, rsik, rsid, t3, adx, adxr, apo, aroondown, aroonup, aroonosc, cci, cmo, dx, macd, \
        macdsignal, macdhist, macdext, macdsignalext, macdhistext, macdfix, macdhistfix, macdsignalfix, mfi, minus_dm, \
        minus_di, mom, plus_di, plus_dm, ppo, roc, rocr, rsi, trix, ultosc, willr, beta, correl, linearreg, \
        linearreg_angle, linearreg_intercept, linearreg_slope, stddev, tsf, var, atr, natr, adosc, upperband, \
        middleband, lowerband, dema, ema, kama, ma, midpoint, midprice, sma, tema, trima, wma = ([] for i in range(
            63))  # counted 63 twice
        for timeperiod in timeperiods:
            benchstart = datetime.utcnow()
            fastperiod = int(timeperiod / 2)
            slowperiod = timeperiod
            stochk = int(timeperiod / 3)
            stochd = int(timeperiod / 4)

            # n =  int(len(close) / slicefactor)
            # offset = (len(close)-1) - (n * slicefactor)
            # opens= open[offset::slicefactor]
            # high= high[offset::slicefactor]
            # low= low[offset::slicefactor]
            # close= close[offset::slicefactor]
            # volumes= volume[offset::slicefactor]

            # now for the weird indicators matype=1 for life, ema
            stoch1_temp, stoch2_temp = ta.STOCH(high, low, close, fastk_period=stochk, slowk_period=stochd,
                                                slowk_matype=1, slowd_period=stochd, slowd_matype=1)
            slowk.append(stoch1_temp)
            slowd.append(stoch2_temp)
            stoch1_temp, stoch2_temp = ta.STOCHF(high, low, close, fastk_period=stochk, fastd_period=stochd,
                                                 fastd_matype=1)
            fastk.append(stoch1_temp)
            fastd.append(stoch2_temp)
            stoch1_temp, stoch2_temp = ta.STOCHRSI(close, timeperiod=timeperiod, fastk_period=stochk,
                                                   fastd_period=stochd, fastd_matype=1)
            rsik.append(stoch1_temp)
            rsid.append(stoch2_temp)
            t3.append(ta.T3(close, timeperiod=timeperiod, vfactor=.7))
            # now for the normal indicators
            # momentum
            adx.append(ta.ADX(high, low, close, timeperiod=timeperiod))
            adxr.append(ta.ADXR(high, low, close, timeperiod=timeperiod))
            apo.append(ta.APO(close, fastperiod=fastperiod, slowperiod=slowperiod))
            aroondown_temp, aroonup_temp = ta.AROON(high, low, timeperiod=timeperiod)
            aroondown.append(aroondown_temp)
            aroonup.append(aroonup_temp)
            aroonosc.append(ta.AROONOSC(high, low, timeperiod=timeperiod))
            cci.append(ta.CCI(high, low, close, timeperiod=timeperiod))
            cmo.append(ta.CMO(close, timeperiod=timeperiod))
            dx.append(ta.DX(high, low, close, timeperiod=timeperiod))
            macd_temp, macdsignal_temp, macdhist_temp = ta.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod,
                                                                signalperiod=stochk)
            macd.append(macd_temp)
            macdsignal.append(macdsignal_temp)
            macdhist.append(macdhist_temp)
            macd_temp, macdsignal_temp, macdhist_temp = ta.MACDEXT(close,
                                                                   fastperiod=fastperiod, fastmatype=1,
                                                                   slowperiod=slowperiod, slowmatype=1,
                                                                   signalperiod=stochk, signalmatype=1)
            macdext.append(macd_temp)
            macdsignalext.append(macdsignal_temp)
            macdhistext.append(macdhist_temp)
            macd_temp, macdsignal_temp, macdhist_temp = ta.MACDFIX(close, signalperiod=stochk)
            macdfix.append(macd_temp)
            macdsignalfix.append(macdsignal_temp)
            macdhistfix.append(macdhist_temp)
            mfi.append(ta.MFI(high, low, close, volume, timeperiod=timeperiod))
            minus_di.append(ta.MINUS_DI(high, low, close, timeperiod=timeperiod))
            minus_dm.append(ta.MINUS_DM(high, low, timeperiod=timeperiod))
            mom.append(ta.MOM(close, timeperiod=timeperiod))
            plus_di.append(ta.PLUS_DI(high, low, close, timeperiod=timeperiod))
            plus_dm.append(ta.PLUS_DM(high, low, timeperiod=timeperiod))
            ppo.append(ta.PPO(close, fastperiod=fastperiod, slowperiod=slowperiod, matype=1))
            roc.append(ta.ROC(close, timeperiod=timeperiod))
            rocr.append(ta.ROCR(close, timeperiod=timeperiod))
            rsi.append(ta.RSI(close, timeperiod=timeperiod))
            trix.append(ta.TRIX(close, timeperiod=timeperiod))
            ultosc.append(
                ta.ULTOSC(high, low, close, timeperiod1=stochk, timeperiod2=fastperiod, timeperiod3=timeperiod))
            willr.append(ta.WILLR(high, low, close, timeperiod=timeperiod))
            # end momentum, begin statistics
            beta.append(ta.BETA(high, low, timeperiod=timeperiod))
            correl.append(ta.CORREL(high, low, timeperiod=timeperiod))
            linearreg.append(ta.LINEARREG(close, timeperiod=timeperiod))
            linearreg_angle.append(ta.LINEARREG_ANGLE(close, timeperiod=timeperiod))
            linearreg_intercept.append(ta.LINEARREG_INTERCEPT(close, timeperiod=timeperiod))
            linearreg_slope.append(ta.LINEARREG_SLOPE(close, timeperiod=timeperiod))
            stddev.append(ta.STDDEV(close, timeperiod=timeperiod, nbdev=1))
            tsf.append(ta.TSF(close, timeperiod=timeperiod))
            var.append(ta.VAR(close, timeperiod=timeperiod, nbdev=1))
            # end stats, begin vol
            atr.append(ta.ATR(high, low, close, timeperiod=timeperiod))
            natr.append(ta.NATR(high, low, close, timeperiod=timeperiod))
            adosc.append(ta.ADOSC(high, low, close, volume, fastperiod=fastperiod, slowperiod=slowperiod))
            # end vol, begin overlaps
            upperband_temp, middleband_temp, lowerband_temp = ta.BBANDS(close, timeperiod=timeperiod, matype=1)
            upperband.append(upperband_temp)
            middleband.append(middleband_temp)
            lowerband.append(lowerband_temp)
            dema.append(ta.DEMA(close, timeperiod=timeperiod))
            ema.append(ta.EMA(close, timeperiod=timeperiod))
            kama.append(ta.KAMA(close, timeperiod=timeperiod))
            ma.append(ta.MA(close, timeperiod=timeperiod, matype=0))
            midpoint.append(ta.MIDPOINT(close, timeperiod=timeperiod))
            midprice.append(ta.MIDPRICE(high, low, timeperiod=timeperiod))
            sma.append(ta.SMA(close, timeperiod=timeperiod))
            tema.append(ta.TEMA(close, timeperiod=timeperiod))
            trima.append(ta.TRIMA(close, timeperiod=timeperiod))
            wma.append(ta.WMA(close, timeperiod=timeperiod))
            # print("finishing timeperiod" + str(timeperiod))
            # print("TA for " + str(timeperiod) + " took " + str(datetime.utcnow() - benchstart))
        # 1 min, 10min, 100m, currentprice, indicators
        if live:
            finishedline = (close[-1])
        else:
            if not flip:
                finishedline = ( # date, flip, 1, 10, 20, 40, 80, 120, current price
                    (end - datetime(1970, 1, 1)) / timedelta(seconds=1), 0, data[-120][4], data[-111][4],
                    data[-101][4], data[-81][4], data[-41][4], data[-1][4], data[-121][4])
            if flip:
                finishedline = ( # date, flip, 1, 10, 20, 40, 80, 120, current price
                    (end - datetime(1970, 1, 1)) / timedelta(seconds=1), 1, data[-120], data[-111],
                    data[-101], data[-81], data[-41], data[-1], data[-121])

        finishedline += (trange, obv, ad, ht_trendline, ht_dcperiod, ht_dcphase,
                         ht_phasor_inphase, ht_phasor_quadrature,
                         ht_sine, ht_leadsine, ht_trendmode, mama, fama, sar, sarext, bop, slowk, slowd, fastk, fastd,
                         rsik, rsid,t3, adx, adxr, apo,
                         aroondown, aroonup, aroonosc, cci, cmo, dx, macd, macdsignal, macdhist, macdext, macdsignalext,
                         macdhistext,
                         macdfix, macdsignalfix, macdhistfix, mfi, minus_di, minus_dm, mom, plus_di, plus_dm, ppo, roc,
                         rocr, rsi, trix,
                         ultosc, willr, beta, correl, linearreg, linearreg_angle, linearreg_intercept, linearreg_slope,
                         stddev, tsf, var,
                         atr, natr, adosc, upperband, middleband, lowerband, dema, ema, kama, ma, midpoint, midprice,
                         sma, tema, trima, wma)
        finishedline = flatten(finishedline)

        if live:
            finishedline = np.array(finishedline, dtype=np.double)
            finishedline = finishedline[~np.isnan(finishedline)]  # remove nans
        else:
            finishedline = np.array(finishedline, dtype=np.double)
            finishedline = finishedline[~np.isnan(finishedline)]
        # print(pid + " has finished")
        if len(finishedline) != 326: break
        return finishedline
        break


def mycallback(x):
    # print(x)
    if x is not None:
        writebuffer.append(x)


def main(ns):
    global writebuffer
    writebuffer = []

    # dbcur.setexectrace(mytrace)


    # offsettime = now - timedelta(weeks=4)
    # data_start = datetime(2015, 1, 30)
    offset = timedelta(weeks=10)
    # offsettime = now -offset
    # # print(now.strftime("%Y-%m-%dT%H:%M:%S"))
    # # print(offsettime.strftime("%Y-%m-%dT%H:%M:%S"))
    # previoustime = offsettime
    globalvar.dbcur.execute("SELECT time FROM futures ORDER BY time DESC LIMIT 1")
    now = datetime.utcfromtimestamp(globalvar.dbcur.fetchone()[0])
    now = now.replace(microsecond=0)
    globalvar.dbcur.execute("SELECT time FROM futures ORDER BY time ASC LIMIT 1")
    data_start = datetime.utcfromtimestamp(globalvar.dbcur.fetchone()[0])
    # print(data_start, now)
    globalvar.l.acquire()
    try:
        globalvar.dbcur.execute(
            "CREATE TABLE IF NOT EXISTS calculated (time INT UNIQUE, flip INT, mainblob BLOB)")
    finally:
        globalvar.l.release()

    for i in range(10000000):
        benchstart = datetime.utcnow()
        pool = multiprocessing.Pool(processes=12)  # todo keep running
        Iterator = []
        for p in range(100):
            Iterator.append(genrandomtime(data_start, now, offset))
        # with open("normalizeddata.csv", 'w') as file:
        # writer = csv.writer(file)
        # print("main thread beginning pool")
        # populatedpool =pool.starmap_async(dataproc_worker, Iterator, callback = mycallback, chunksize=4000)
        # globalvar.l.acquire()
        for argument in Iterator:
        #
        #
        #     globalvar.dbcur.execute("SELECT * FROM futures WHERE time BETWEEN (?) AND (?)",
        #                             [(argument[0] - datetime(1970, 1, 1)) / timedelta(seconds=1),
        #                              (argument[1] - datetime(1970, 1, 1)) / timedelta(seconds=1)])
        #     data = globalvar.dbcur.fetchall()

            argument += (False, ns)
            pool.apply_async(dataproc_worker, argument, callback=mycallback)
        # globalvar.l.release()
        pool.close()
        pool.join()
        try:
            get_history.main(ns)
        except:
            pass
        #    while populatedpool.ready() == False:
        #        sleep(1)
        #
        # file = open("normalizeddata.csv", 'a')
        # writer = csv.writer(file)
        #
        # for row in writebuffer:
        #     if row != None:
        #         writer.writerow(row)

        # insert to DB
        globalvar.l.acquire()
        try:
            globalvar.dbcur.execute('BEGIN TRANSACTION')
            for line in writebuffer:
                globalvar.dbcur.execute("INSERT OR REPLACE INTO calculated VALUES (?,?,?)",
                                        [int(line[0]), line[1], line])
            globalvar.dbcur.execute('COMMIT')
        finally:
            globalvar.l.release()
        writebuffer = []  # reset for next loop
        print("Iteration " + str(i) + " took " + str(datetime.utcnow() - benchstart), "for 100 samples")


# def oldfunctionthatdoesnothing:
#     timeline = dbcur.fetchone()
#     for value in timeline:
#         time = datetime.strptime(value[0], "%Y-%m-%dT%H:%M:%SZ")
#         timestr = value[0]
#         if time - previoustime > timedelta(minutes=2) and previoustime != offsettime:
#             timelag = time - previoustime
#             # print(previoustime, time, timelag)
#
#         previoustime = time

if __name__ == '__main__':
    m = Manager()
    ns = m.Namespace()

    main(ns)
