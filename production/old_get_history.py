import codecs
import json
from datetime import datetime, timedelta
from urllib.request import urlopen
from multiprocessing import Manager
import globalvar


# possibilities = list(permutations(['futures','ticker','okcoin','btcusd','quarter'], 4))
#
#
# for permute in possibilities:
#     try:
#         url = "https://cryptoiq.io/api/marketdata/" + permute[0] + "/" + permute[1] + "/" + permute[2] + "/" + permute[3] + "/2016-03-25/23?sac=E4cvdq2jgDjgk3rW"
#         print(url)
#         response = urlopen(url)
#         reader = codecs.getreader('utf-8')
#         data = json.load(reader(response))
#         print (url)
#         if url == "https://cryptoiq.io/api/marketdata/ticker/okcoin/btcusd/2016-03-25/23?sac=E4cvdq2jgDjgk3rW":
#             continue
#         break
#     except:
#         continue


def date_range(start=None, end=None):
    span = end - start
    for i in range(int(span.seconds/3600)+int(span.days*24)+2):
        yield start + timedelta(seconds=3600*i)


# url = "https://cryptoiq.io/api/marketdata/futures/ticker/okcoin/btcusd/quarter/2016-03-30/23"

# consider using LMDB instead of SQLite
# okcoincny start date = 2015-01-30
def main(ns):


    today = datetime.utcnow()
    # today = date.isoformat(today)
    # print (today.__str__())
    globalvar.l.acquire()
    try:
        globalvar.dbcur.execute("CREATE TABLE IF NOT EXISTS spot (time TEXT UNIQUE, last REAL, high REAL, low REAL, volume REAL )")
        globalvar.dbcur.execute("SELECT MAX(time) FROM spot")
        d1 = globalvar.dbcur.fetchone()[0]
    finally:
        globalvar.l.release()
    d1 = datetime.strptime(d1, "%Y-%m-%dT%H:%M:%SZ")
    d1 = d1-timedelta(hours=1)
    # d1 = datetime(2016, 11, 6)  # uncomment to get from #date
    print(d1, today)
    #d1 = datetime(2016, 9, 1)
    # memdb =apsw.Connection(":memory:")
    # with memdb.backup("main", db, "main") as backup:
    #    backup.step() #copy db to memory
    # memcursor = memdb.cursor()
    for hour in date_range(start=d1, end=today):
        # print (date.strftime("%Y-%m-%d"))


        url = "https://cryptoiq.io/api/marketdata/ticker/okcoin/btccny/" + hour.strftime("%Y-%m-%d") + "/" \
              + hour.strftime("%H")
        #print(url)
        response = urlopen(url)
        reader = codecs.getreader('utf-8')
        data = json.load(reader(response))
        print("processing spot " + hour.strftime("%Y-%m-%d") + " hour " + hour.strftime("%H") + " Lines:" + str(len(data)))
        # print (data)  #debug command

        # testtime = data[0]['time']
        # testtime2 = data[20]['time']

        # print (data[0]['time'],data[0]['ask'],data[0]['bid'],data[0]['last'])
        # for line in data:   #convert 2015-03-30T23:00:00Z to 2015-03-30 23:00:00
        #    badtime=line['time']
        globalvar.l.acquire()
        try:
            globalvar.dbcur.execute('BEGIN TRANSACTION')
            for line in data:
                globalvar.dbcur.execute("INSERT OR IGNORE INTO spot VALUES (?,?,?,?,?)",
                              [line['time'], line['last'], line['high'], line['low'], line['volume']])
            globalvar.dbcur.execute('COMMIT')
        finally:
            globalvar.l.release()
            # memdb.commit()

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


def fix_gap(datetime):
    url = "https://cryptoiq.io/api/marketdata/ticker/okcoin/btccny/" + datetime.strftime("%Y-%m-%d") \
          + "/" + datetime.strftime("%H").zfill(2)
    response = urlopen(url)
    reader = codecs.getreader('utf-8')
    data = json.load(reader(response))
    print("processing spot " + datetime.strftime("%Y-%m-%d hour %H") + " Lines:" + str(len(data)))
    # print (data)  #debug command

    # testtime = data[0]['time']
    # testtime2 = data[20]['time']

    # print (data[0]['time'],data[0]['ask'],data[0]['bid'],data[0]['last'])
    # for line in data:   #convert 2015-03-30T23:00:00Z to 2015-03-30 23:00:00
    #    badtime=line['time']
    globalvar.dbcur.execute('BEGIN TRANSACTION')
    for line in data:
        globalvar.dbcur.execute("INSERT OR IGNORE INTO spot VALUES (?,?,?,?,?)",
                      [line['time'], line['last'], line['high'], line['low'], line['volume']])
    globalvar.dbcur.execute('COMMIT')

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
    m = Manager()
    ns = m.Namespace()

    main(ns)
