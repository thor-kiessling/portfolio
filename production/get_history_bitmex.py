import codecs
import json
from datetime import datetime, timedelta, timezone
#from urllib.request import urlopen
#from multiprocessing import Manager
import globalvar
#import OkcoinRestFutureAPI
from dateutil import parser
import apsw
from time import sleep
from requests.exceptions import HTTPError
import api_keys

from market_maker import bitmex

# https://www.bitmex.com/api/explorer/#!/Trade/Trade_getBucketed look at this to refactor to bitmex




def date_range(start=None, end=None):
    span = end - start
    for i in range(int(span.seconds/3600)+int(span.days*24)+2):
        yield start + timedelta(seconds=3600*i)


# url = "https://cryptoiq.io/api/marketdata/futures/ticker/okcoin/btcusd/quarter/2016-03-30/23"

# consider using LMDB instead of SQLite
# okcoincny start date = 2015-01-30
def main(bm):
    getfromfile = False
    symbol = "XBTUSD"

    today = datetime.now(timezone.utc)
    # today = date.isoformat(today)
    # print (today.__str__())

    globalvar.l.acquire()
    try:
        globalvar.dbcur.execute("CREATE TABLE IF NOT EXISTS bmxswap (time INT UNIQUE, open REAL, high REAL, low REAL, close REAL, volume INT )")
        globalvar.dbcur.execute("SELECT MAX(time) FROM bmxswap")
        d1 = globalvar.dbcur.fetchone()[0]
    finally:
        globalvar.l.release()
    #d1_date = datetime.utcfromtimestamp(d1)
    today_stamp = int(datetime.timestamp(today))
    epoch_timestamp = datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp()  # uncomment to get from #date
    if d1 is None:
        d1 = epoch_timestamp
    #timediff = ((today_stamp - d1) / 60) +1
    #timediff = round(timediff +.5)
    #if timediff < 0: timediff = 500
    start = round((d1 - epoch_timestamp) / 60)
    try:
        data = bm.historical_ohlc(symbol=symbol, start=start, count=500)
    except HTTPError:
        sleep(1)
        return False

    done = False
    if len(data) < 500:
        done = True

    globalvar.l.acquire()
    try:
        globalvar.dbcur.execute('BEGIN TRANSACTION')
        for line in data:
            globalvar.dbcur.execute("INSERT OR REPLACE INTO bmxswap VALUES (?,?,?,?,?,?)",
                          [parser.parse(line['timestamp']).timestamp(), line['open'], line['high'],
                           line['low'], line['close'], line['volume']]) # using volume in contracts
        globalvar.dbcur.execute('COMMIT')
        # print("success writing history to data")
    except apsw.BusyError:
        sleep(1)
        globalvar.dbcur.execute('COMMIT')
        main(bm)
    finally:
        globalvar.l.release()
        # memdb.commit()
    return done

#    dbcur.execute("CREATE TABLE IF NOT EXISTS spotorderbook (time TEXT UNIQUE, asks BLOB, bids BLOB)")
#
#    #d1 = datetime(2016, 6, 28)
#    for date in date_range(start=d1, end=today):
#        # print (date.strftime("%Y-%m-%d"))
#        i = 0
#        while i <= 23:
#
#            url = "https://cryptoiq.io/api/marketdata/orderbooktop/okcoin/btccny/" + date.strftime(
#                "%Y-%m-%d") + "/" + str(i).zfill(2) + "?sac=E4cvdq2jgDjgk3rW"
#            response = urlopen(url)
#            reader = codecs.getreader('utf-8')
#            data = json.load(reader(response))
#            print("processing spot order book " + date.strftime("%Y-%m-%d") + " hour " + str(i) + " Lines:" + str(
#                len(data)))
            # print (data)  #debug command

            # testtime = data[0]['time']
            # testtime2 = data[20]['time']

            # print (data[0]['time'],data[0]['ask'],data[0]['bid'],data[0]['last'])
            # for line in data:   #convert 2015-03-30T23:00:00Z to 2015-03-30 23:00:00
            #    badtime=line['time']
#            dbcur.execute('BEGIN TRANSACTION')
 #           for line in data:
  #              asks = json.dumps(line['asks'])
  #              bids = json.dumps(line['bids'])
  #              dbcur.execute("INSERT OR IGNORE INTO spotorderbook VALUES (?,?,?)",
  #                            [line['time'], asks, bids])
   #         dbcur.execute('COMMIT')
            # memdb.commit()
    #        i = i + 1


# def fix_gap(datetime):
#     url = "https://cryptoiq.io/api/marketdata/ticker/okcoin/btccny/" + datetime.strftime("%Y-%m-%d") \
#           + "/" + datetime.strftime("%H").zfill(2)
#     response = urlopen(url)
#     reader = codecs.getreader('utf-8')
#     data = json.load(reader(response))
#     print("processing spot " + datetime.strftime("%Y-%m-%d hour %H") + " Lines:" + str(len(data)))
#     #print (data)  #debug command
#
#     # testtime = data[0]['time']
#     # testtime2 = data[20]['time']
#
#     # print (data[0]['time'],data[0]['ask'],data[0]['bid'],data[0]['last'])
#     # for line in data:   #convert 2015-03-30T23:00:00Z to 2015-03-30 23:00:00
#     #    badtime=line['time']
#     globalvar.dbcur.execute('BEGIN TRANSACTION')
#     for line in data:
#         globalvar.dbcur.execute("INSERT OR IGNORE INTO spot VALUES (?,?,?,?,?)",
#                       [line['time'], line['last'], line['high'], line['low'], line['volume']])
#     globalvar.dbcur.execute('COMMIT')

#
# data = 0
# start =datetime.now()
# testlist = dbcur.execute('SELECT * FROM spot ORDER BY time')
# end = datetime.now()
# print("select: " + str(end-start))
# start =datetime.now()
# i = 0
# for row in dbcur.execute('SELECT * FROM spot ORDER BY time'):
#     i = i + 1
#     # timebench: 37.784652s SQLite on 26m rows 0 mmap
# end = datetime.now()
# print(i)
# print("select+ count: "+ str(end-start))
#
# i=0
# start =datetime.now()
# while i < 26006683:
#     i = i+1
# end = datetime.now()
# print("counting:" + str(end-start))
#
# db.close()

if __name__ == "__main__":
    #m = Manager()
    #ns = m.Namespace()
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
    symbol = "XBTUSD"
    done = False
    while done is False:
        done = main(bm)
        print("finished a run")

