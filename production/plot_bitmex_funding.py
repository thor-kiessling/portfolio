import globalvar
import numpy as np
from matplotlib import pyplot as plt
import get_fund_history_bitmex
import get_history_bitmex
from market_maker import bitmex
import api_keys

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

print("loading from DB")
globalvar.dbcur.execute("SELECT * FROM bmxswap ORDER BY time ASC")
data = globalvar.dbcur.fetchall()
# dt = np.dtype(np.double)
print("converting from DB")
data = np.array(data)

print("loading from DB")
globalvar.dbcur.execute("SELECT * FROM bmxfund ORDER BY time ASC")
data2 = globalvar.dbcur.fetchall()
# dt = np.dtype(np.double)
print("converting from DB")
data2 = np.array(data2)

data = data[-172000:, :]



mediana = []
medianb = []
median1 = []
median2 = []
median3 = []
median4 = []
median5 = []
median6 = []
median7 = []
median8 = []
startindex = None
for index1, line1 in enumerate(data):
    if startindex is not None: break
    for line2 in data2:
        difference = line1[0] - line2[0]
        if line1[0] == line2[0]:
            startindex = index1
            break


for index1, line1 in enumerate(data[startindex::480]):
    index1 = (index1 * 480) + startindex
    for line2 in data2:
        if line1[0] == line2[0]:  # this matches the minute directly before funding, use close prices
            if abs(line2[1]) > .0030:
                if line2[1] > 0:
                    temp_deub_var = data[index1 - 30: index1 + 20]
                    append_data = data[index1 - 30: index1 + 20, 4] / data[index1, 4]
                    mediana.append(append_data)
                    break
                else:
                    medianb.append(data[index1 - 30: index1 + 20, 4] / data[index1, 4])
                    break
            elif abs(line2[1]) > .0020:
                if line2[1] > 0:
                    median1.append(data[index1 - 30: index1 + 20, 4] / data[index1, 4])
                    break
                else:
                    median2.append(data[index1 - 30: index1 + 20, 4] / data[index1, 4])
                    break
            elif abs(line2[1]) > .0015:
                if line2[1] > 0:
                    median3.append(data[index1 - 30: index1 + 20, 4] / data[index1, 4])
                    break
                else:
                    median4.append(data[index1 - 30: index1 + 20, 4] / data[index1, 4])
                    break
            elif abs(line2[1]) > .0010:
                if line2[1] > 0:
                    median5.append(data[index1 - 30: index1 + 20, 4] / data[index1, 4])
                    break
                else:
                    median6.append(data[index1 - 30: index1 + 20, 4] / data[index1, 4])
                    break
            elif abs(line2[1]) > .0005:
                if line2[1] > 0:
                    median7.append(data[index1 - 30: index1 + 20, 4] / data[index1, 4])
                    break
                else:
                    median8.append(data[index1 - 30: index1 + 20, 4] / data[index1, 4])
                    break
    if index1 % 100 == 0:
        print(index1/len(data))

list_of_runs = [[mediana, .003], [medianb, -.003], [median1, .002],[median2, -.002], [median3, .0015],
                [median4, -.0015], [median5, .001], [median6, -.001], [median7, .0005], [median8, -.0005]]
for run in list_of_runs:
    amount = run[1]
    run = run[0]
    # print(len(run), "runs for", amount)
    margincall = .01
    try:
        runs_margin_calls = []
        runs_returns = []
        play_postfunding_outer = []
        for begin_index in range(30):
            no_margin_calls = 0
            returns = []
            play_postfunding_tally = 0
            for single_run in run:
                beginning_price = None
                for step, price in enumerate(single_run):
                    if step == begin_index:
                        beginning_price = price
                        play_postfunding = False
                    if not beginning_price: continue
                    if step == 30:  # item 30 is the minute before funding. Use the close price
                        price30 = price
                        if amount > 0:  #short
                            returns.append(beginning_price - price)
                        else:
                            returns.append(price - beginning_price)

                    if amount > 0 and step < 30: # if short, remember inverse the fund rate
                        if price > beginning_price + margincall:
                            no_margin_calls += 1
                            returns.append(beginning_price - price)
                    elif step < 30:
                        if price < beginning_price - margincall:
                            no_margin_calls += 1
                            returns.append(price - beginning_price)

                    if step > 30 and step < 35:
                        if amount > 0: # short prefunding, now long postfunding
                            if price > price30 + .0005:  # confirmed funding %
                                play_postfunding = True
                        else:
                            if price < price30 - .0005:
                                play_postfunding = True
                if play_postfunding:
                    play_postfunding_tally += 1





            margin_calls = round(no_margin_calls/len(run) * 100, 1)
            returns_cumulative = int((np.mean(returns) + abs(amount/2)) * 100 * 45)
            play_postfunding_outer.append(True if (play_postfunding_tally > len(run)*.85) else False)
            runs_margin_calls.append(margin_calls)
            runs_returns.append(returns_cumulative)
            # print("at T-" + str(30-begin_index)+ " there were " + str(margin_calls) +
            #       "% margin calls for " + str(amount) + " and returns of " + str(returns_cumulative))
        # print(runs_margin_calls)
        # print(runs_returns)
        bsf = [-1, -100, False]
        for index, margin_call in enumerate(runs_margin_calls):
            returns = runs_returns[index]
            play_postfunding = play_postfunding_outer[index]
            if margin_call > 2:
                bsf = [-1, -100, False]
            if margin_call < 2 and returns > 0 and returns > bsf[1]:
                bsf = [index, returns, play_postfunding]
        if bsf[0] != -1:
            print("for", amount, "use T-" + str(30-bsf[0]), "for", bsf[1], "% returns", "play the post-funding:", play_postfunding)
        else:
            print("dont play prefunding for", amount, "play the post-funding:", play_postfunding)
    except ZeroDivisionError:
        print("No runs for" + str(amount))
        continue


#                                            rgb color triad floats
if len(mediana) > 0:
    plt.plot(range(50), np.median(mediana, axis=0), color=(0, 1, 0), label="30 long")
if len(medianb) > 0:
    plt.plot(range(50), np.median(medianb, axis=0), color=(1, 0, 0), label='30 short')
if len(median1) > 0:
    plt.plot(range(50), np.median(median1, axis=0), color=(0, .8, 0), label="20 long")
if len(median2) > 0:
    plt.plot(range(50), np.median(median2, axis=0), color=(.8, 0, 0), label='20 short')
if len(median3) > 0:
    plt.plot(range(50), np.median(median3, axis=0), color=(0, .6, 0), label="15 long")
if len(median4) > 0:
    plt.plot(range(50), np.median(median4, axis=0), color=(.6, 0, 0), label='15 short')
if len(median5) > 0:
    plt.plot(range(50), np.median(median5, axis=0), color=(0, .4, 0), label="10 long")
if len(median6) > 0:
    plt.plot(range(50), np.median(median6, axis=0), color=(.4, 0, 0), label='10 short')
# if len(median7) > 0:
#     plt.plot(range(50), np.median(median7, axis=0), color=(0, .2, 0), label="5 long")
# if len(median8) > 0:
#     plt.plot(range(50), np.median(median8, axis=0), color=(.2, 0, 0), label='5 short')
# if len(mediana) > 0:
#     plt.plot(range(50), np.mean(mediana, axis=0), color=(0, 1, 0), label="30 long")
# if len(medianb) > 0:
#     plt.plot(range(50), np.mean(medianb, axis=0), color=(1, 0, 0), label='30 short')
# if len(median1) > 0:
#     plt.plot(range(50), np.mean(median1, axis=0), color=(0, .8, 0), label="20 long")
# if len(median2) > 0:
#     plt.plot(range(50), np.mean(median2, axis=0), color=(.8, 0, 0), label='20 short')
# if len(median3) > 0:
#     plt.plot(range(50), np.mean(median3, axis=0), color=(0, .6, 0), label="15 long")
# if len(median4) > 0:
#     plt.plot(range(50), np.mean(median4, axis=0), color=(.6, 0, 0), label='15 short')
# if len(median5) > 0:
#     plt.plot(range(50), np.mean(median5, axis=0), color=(0, .4, 0), label="10 long")
# if len(median6) > 0:
#     plt.plot(range(50), np.mean(median6, axis=0), color=(.4, 0, 0), label='10 short')
# if len(median7) > 0:
#     plt.plot(range(50), np.mean(median7, axis=0), color=(0, .2, 0), label="5 long")
# if len(median8) > 0:
#     plt.plot(range(50), np.mean(median8, axis=0), color=(.2, 0, 0), label='5 short')
plt.legend()


plt.show()

# 003   t-4
# 002   t-11 12
# 0015  t-7
# 0010  t-15
# 0005  t-1 1 # don't even do this
# -0005 t-6
# -0010  t-3
# -0015 t-6
# -002  t-6
# -003 t-10
