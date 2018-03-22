"""BitMEX API Connector."""
from __future__ import absolute_import
import requests
import time
import datetime
import json
import base64
import uuid
import logging
import urllib3
from operator import itemgetter
from time import sleep
from market_maker.auth import AccessTokenAuth, APIKeyAuthWithExpires
from market_maker.utils import constants, errors
from market_maker.ws.ws_thread import BitMEXWebsocket


# https://www.bitmex.com/api/explorer/
class BitMEX(object):

    """BitMEX API Connector."""

    def __init__(self, base_url=None, symbol=None, login=None, password=None, otpToken=None,
                 apiKey=None, apiSecret=None, orderIDPrefix='mm_bitmex_', shouldWSAuth=True):
        """Init connector."""
        self.logger = logging.getLogger('root')
        self.base_url = base_url
        self.symbol = symbol
        self.token = None
        # User/pass auth is no longer supported
        if (login or password or otpToken):
            raise Exception("User/password authentication is no longer supported via the API. Please use " +
                            "an API key. You can generate one at https://www.bitmex.com/app/apiKeys")
        if (apiKey is None):
            raise Exception("Please set an API key and Secret to get started. See " +
                            "https://github.com/BitMEX/sample-market-maker/#getting-started for more information."
                            )
        self.apiKey = apiKey
        self.apiSecret = apiSecret
        if len(orderIDPrefix) > 13:
            raise ValueError("settings.ORDERID_PREFIX must be at most 13 characters long!")
        self.orderIDPrefix = orderIDPrefix

        # Prepare HTTPS session
        self.session = requests.Session()
        # These headers are always sent
        self.session.headers.update({'user-agent': 'liquidbot-' + constants.VERSION})
        self.session.headers.update({'content-type': 'application/json'})
        self.session.headers.update({'accept': 'application/json'})

        # Create websocket for streaming data
        self.ws = BitMEXWebsocket()
        self.ws.connect(base_url, symbol, shouldAuth=shouldWSAuth, secret=apiSecret, key=apiKey)

    def __del__(self):
        self.exit()

    def exit(self):
        self.ws.exit()

    def reconnect(self):
        self.logger.info("is alive"+str(self.ws.wst.is_alive()))
        self.logger.info("exited?"+str(self.ws.exited))
        self.ws.exit()
        self.ws = BitMEXWebsocket()
        self.logger.info("Reconnecting to websocket")
        self.ws.connect(self.base_url, self.symbol, shouldAuth=True, secret=self.apiSecret, key=self.apiKey)

    #
    # Public methods
    #
    def ticker_data(self, symbol):
        """Get ticker data."""
        return self.ws.get_ticker(symbol)

    def instrument(self, symbol):
        """Get an instrument's details."""
        return self.ws.get_instrument(symbol)

    def market_depth(self, symbol):
        """Get market depth / orderbook."""
        return self.ws.market_depth(symbol)

    def best_bid_ask(self, symbol='XBTUSD'):
        # orderbook = self.market_depth(symbol)
        # #orderbook = sorted(orderbook, key=lambda k: -k['price'])
        # orderbook = sorted(orderbook, key=itemgetter('price'), reverse=True)
        # if orderbook is None or len(orderbook) is 0:
        #     sleep(.1)
        #     return self.best_bid_ask()
        # best_bid = orderbook[-1]['price']
        # best_ask = orderbook[0]['price']
        # for ob_entry in orderbook:
        #     if ob_entry['side'] == 'Sell' and ob_entry['price'] < best_ask:
        #         best_ask = ob_entry['price']
        #     if ob_entry['side'] == 'Buy' and ob_entry['price'] > best_bid:
        #         best_bid = ob_entry['price']
        return self.ws.best_bid_ask()

    def recent_trades(self, symbol):
        """Get recent trades.

        Returns
        -------
        A list of dicts:
              {u'amount': 60,
               u'date': 1306775375,
               u'price': 8.7401099999999996,
               u'tid': u'93842'},

        """
        return self.ws.recent_trades(symbol)

    #
    # Authentication required methods
    #
    def authentication_required(function):
        """Annotation for methods that require auth."""
        def wrapped(self, *args, **kwargs):
            if not (self.apiKey):
                msg = "You must be authenticated to use this method"
                raise errors.AuthenticationError(msg)
            else:
                return function(self, *args, **kwargs)
        return wrapped

    @authentication_required
    def funds(self):
        """Get your current balance."""
        return self.ws.funds()

    @authentication_required
    def position(self, symbol):
        """Get your open position."""
        return self.ws.position(symbol)

    @authentication_required
    def buy(self, quantity, price):
        """Place a buy order.

        Returns order object. ID: orderID
        """
        return self.place_order(quantity, price)

    @authentication_required
    def sell(self, quantity, price):
        """Place a sell order.

        Returns order object. ID: orderID
        """
        return self.place_order(-quantity, price)

    @authentication_required
    def place_order(self, quantity, price=None, exec_inst='ParticipateDoNotInitiate'):
        """Place an order."""
        # if price < 0 and not None:
        #     raise Exception("Price must be positive.")

        endpoint = "order"
        # Generate a unique clOrdID with our prefix so we can identify it.
        clOrdID = self.orderIDPrefix + base64.b64encode(uuid.uuid4().bytes).decode('utf-8').rstrip('=\n')
        postdict = {
            'symbol': self.symbol,
            'orderQty': quantity,
            'clOrdID': clOrdID
        }
        if price:
            postdict['price'] = price
        else:
            postdict['ordType'] = 'Market'

        if exec_inst:
            postdict['execInst'] = exec_inst

        return self._curl_bitmex(api=endpoint, postdict=postdict, verb="POST")

    @authentication_required
    def amend_bulk_orders(self, orders):
        """Amend multiple orders."""
        return self._curl_bitmex(api='order/bulk', postdict={'orders': orders}, verb='PUT')

    @authentication_required
    def amend_order(self, order):
        """Amend multiple orders."""
        return self._curl_bitmex(api='order', postdict=order, verb='PUT')

    @authentication_required
    def create_bulk_orders(self, orders):
        """Create multiple orders."""
        for order in orders:
            order['clOrdID'] = self.orderIDPrefix + base64.b64encode(uuid.uuid4().bytes).decode('utf-8').rstrip('=\n')
            order['symbol'] = self.symbol
        return self._curl_bitmex(api='order/bulk', postdict={'orders': orders}, verb='POST')

    @authentication_required
    def open_orders(self):
        """Get open orders."""
        return self.ws.open_orders(self.orderIDPrefix)

    @authentication_required
    def http_open_orders(self):
        """Get open orders via HTTP. Used on close to ensure we catch them all."""
        api = "order"
        orders = self._curl_bitmex(
            api=api,
            query={'filter': json.dumps({'ordStatus.isTerminated': False, 'symbol': self.symbol})},
            verb="GET"
        )
        # Only return orders that start with our clOrdID prefix.
        return [o for o in orders if str(o['clOrdID']).startswith(self.orderIDPrefix)]

    @authentication_required
    def cancel(self, orderID):
        """Cancel an existing order."""
        api = "order"
        postdict = {
            'orderID': orderID,
        }
        return self._curl_bitmex(api=api, postdict=postdict, verb="DELETE")

    @authentication_required
    def withdraw(self, amount, fee, address):
        api = "user/requestWithdrawal"
        postdict = {
            'amount': amount,
            'fee': fee,
            'currency': 'XBt',
            'address': address
        }
        return self._curl_bitmex(api=api, postdict=postdict, verb="POST")

    @authentication_required
    def historical_ohlc(self, symbol, start, count):
        api = "trade/bucketed"
        postdict = {
            'symbol': symbol,
            'binSize': "1m",
            'count': count,
            'start': start
        }
        return self._curl_bitmex(api=api, postdict=postdict, verb="GET")

    @authentication_required
    def historical_funding(self, symbol, start, count):
        api = "funding"
        postdict = {
            'symbol': symbol,
            'count': count,
            'start': start
        }
        return self._curl_bitmex(api=api, postdict=postdict, verb="GET")

    def _curl_bitmex(self, api, query=None, postdict=None, timeout=20, verb=None):
        """Send a request to BitMEX Servers."""
        # Handle URL
        url = self.base_url + api

        # Default to POST if data is attached, GET otherwise
        if not verb:
            verb = 'POST' if postdict else 'GET'

        # Auth: Use Access Token by default, API Key/Secret if provided
        auth = AccessTokenAuth(self.token)
        if self.apiKey:
            auth = APIKeyAuthWithExpires(self.apiKey, self.apiSecret)

        # Make the request
        try:
            req = requests.Request(verb, url, json=postdict, auth=auth, params=query)
            prepped = self.session.prepare_request(req)
            response = self.session.send(prepped, timeout=timeout)
            # Make non-200s throw
            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            # 401 - Auth error. This is fatal with API keys.
            if response.status_code == 401:
                self.logger.error("Login information or API Key incorrect, please check and restart.")

            # 404, can be thrown if order canceled does not exist.
            elif response.status_code == 404:
                if verb == 'DELETE':
                    self.logger.error("Order not found: %s" % postdict['orderID'])
                    return
                self.logger.error("Unable to contact the BitMEX API (404). ")

            # 429, ratelimit; cancel orders & wait until X-Ratelimit-Reset
            elif response.status_code == 429:
                self.logger.error("Ratelimited on current request. Sleeping, then trying again. Try fewer " +
                                  "order pairs or contact support@bitmex.com to raise your limits. " +
                                  "Request: %s \n %s" % (url, json.dumps(postdict)))

                # Figure out how long we need to wait.
                ratelimit_reset = response.headers['X-Ratelimit-Reset']
                to_sleep = int(ratelimit_reset) - int(time.time())
                reset_str = datetime.datetime.fromtimestamp(int(ratelimit_reset)).strftime('%X')

                # We're ratelimited, and we may be waiting for a long time. Cancel orders.
                self.logger.warning("Canceling all known orders in the meantime.")
                self.cancel([o['orderID'] for o in self.open_orders()])

                self.logger.error("Your ratelimit will reset at %s. Sleeping for %d seconds." % (reset_str, to_sleep))
                time.sleep(to_sleep)

                # Retry the request.
                return self._curl_bitmex(api, query, postdict, timeout, verb)

            # 503 - BitMEX temporary downtime, likely due to a deploy. Try again
            elif response.status_code == 503:
                self.logger.warning("Unable to contact the BitMEX API (503), retrying. " +
                                    "Request: %s \n %s" % (url, json.dumps(postdict)))
                time.sleep(3)
                return self._curl_bitmex(api, query, postdict, timeout, verb)

            elif response.status_code == 400:
                error = response.json()['error']
                message = error['message'].lower()
                # Duplicate clOrdID: that's fine, probably a deploy, go get the order and return it
                if 'duplicate clordid' in message:

                    order = self._curl_bitmex('/order',
                                              query={'filter': json.dumps({'clOrdID': postdict['clOrdID']})},
                                              verb='GET')[0]
                    if (
                            order['orderQty'] != abs(postdict['orderQty']) or
                            order['side'] != ('Buy' if postdict['orderQty'] > 0 else 'Sell') or
                            # order['price'] != postdict['price'] or  # no longer checking price because market orders
                            order['symbol'] != postdict['symbol']):
                        raise Exception('Attempted to recover from duplicate clOrdID, but order returned from API ' +
                                        'did not match POST.\nPOST data: %s\nReturned order: %s' % (
                                            json.dumps(postdict), json.dumps(order)))
                    # All good
                    return order
                elif 'insufficient available balance' in message and 'ordType' not in postdict:
                    self.logger.info("Attempt to place order failed due to low balance")
                    self.logger.info("most likely a double spend attempt")
                    self.logger.info(str(postdict))
                    self.logger.info(str(message))
                    #raise Exception('Account out of funds. The message: %s' % error['message'])
                    return None
                elif 'insufficient available balance' in message and postdict['ordType'] == 'Market':
                    self.logger.info("Attempt to place market order failed due low available balance")
                    self.logger.info("stepping down and replacing order")
                    #self.logger.info(str(postdict))
                    #self.logger.info(str(message))
                    #raise Exception('Account out of funds. The message: %s' % error['message'])
                    postdict['clOrdID'] = self.orderIDPrefix + base64.b64encode(uuid.uuid4().bytes).decode('utf-8').rstrip('=\n')
                    postdict['orderQty'] = int(postdict['orderQty'] * .85)
                    return self._curl_bitmex(api, query, postdict, timeout, verb)
                elif 'invalid ordstatus' in message and prepped.method == 'PUT':
                    self.logger.info("Attempt to amend order " + postdict['orderID'] + " failed")
                    return


            # If we haven't returned or re-raised yet, we get here.
            self.logger.error("Error: %s: %s" % (e, response.text))
            self.logger.error("Endpoint was: %s %s: %s" % (verb, api, json.dumps(postdict)))
            raise e

        except requests.exceptions.Timeout or urllib3.exceptions.ReadTimeoutError as e:
            # Timeout, re-run this request
            self.logger.warning("Timed out, retrying...")
            sleep(.1)
            self.reconnect()
            return self._curl_bitmex(api, query, postdict, timeout, verb)

        except requests.exceptions.ConnectionError as e:
            self.logger.warning("Unable to contact the BitMEX API (ConnectionError). Please check the URL. Retrying. " +
                                "Request: %s \n %s" % (url, json.dumps(postdict)))
            time.sleep(1)
            return self._curl_bitmex(api, query, postdict, timeout, verb)

        return response.json()
