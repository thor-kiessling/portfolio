# !/usr/bin/env python

import pprint
import json

from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient

from BitMEXAPIKeyAuthenticator import APIKeyAuthenticator
import api_keys
#
# from market_maker import market_maker   # testing the market maker in it's entirety.
# market_maker.run()

HOST = "https://bitmex.com"
SPEC_URI = HOST + "/api/explorer/swagger.json"

# See full config options at http://bravado.readthedocs.io/en/latest/configuration.html
config = {
    # Don't use models (Python classes) instead of dicts for #/definitions/{models}
    'use_models': True,
    # This library has some issues with nullable fields
    'validate_responses': False,
    # Returns response in 2-tuple of (body, response); if False, will only return body
    'also_return_response': True,
}

bitMEX = SwaggerClient.from_url(
    SPEC_URI,
    config=config)

pp = pprint.PrettyPrinter(indent=2)

# You can get a feel for what is available by printing these objects.
# See also https://testnet.bitmex.com/api/explorer
print('---The BitMEX Object:---')
print(dir(bitMEX))
print('\n---The BitMEX.everything Objects:---')
print(dir(bitMEX.APIKey))
print(dir(bitMEX.Announcement))
print(dir(bitMEX.Chat))
print(dir(bitMEX.Execution))
print(dir(bitMEX.Funding))
print(dir(bitMEX.Instrument))
print(dir(bitMEX.Insurance))
print(dir(bitMEX.Leaderboard))
print(dir(bitMEX.Liquidation))
print(dir(bitMEX.Notification))
print(dir(bitMEX.Order))
print(dir(bitMEX.OrderBook))
print(dir(bitMEX.Position))
print(dir(bitMEX.Quote))
print(dir(bitMEX.Schema))
print(dir(bitMEX.Settlement))
print(dir(bitMEX.Stats))
print(dir(bitMEX.Trade))
print(dir(bitMEX.User))

# for obj in dir(bitMEX):
#     print(dir(bitMEX.obj))

# Basic unauthenticated call
res, http_response = bitMEX.Trade.Trade_get(symbol='XBTUSD', count=1).result()
print('\n---A basic Trade GET:---')
pp.pprint(res)
print('\n---Response details:---')
print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))

#
# Authenticated calls
#
# To do authentication, you must generate an API key.
# Do so at https://testnet.bitmex.com/app/apiKeys

API_KEY = api_keys.bmx_api_key
API_SECRET = api_keys.bmx_api_secret

request_client = RequestsClient()
request_client.authenticator = APIKeyAuthenticator(HOST, API_KEY, API_SECRET)

bitMEXAuthenticated = SwaggerClient.from_url(
    SPEC_URI,
    config=config,
    http_client=request_client)
print(dir(bitMEXAuthenticated))
print(dir(bitMEXAuthenticated.Position))

# Basic authenticated call
print('\n---A basic Position then funding then user deposit addr GET:---')
res, http_response = bitMEXAuthenticated.Position.Position_get().result()
pp.pprint(res)
pp.pprint(http_response.status_code)
res, http_response = bitMEXAuthenticated.Funding.Funding_get(symbol='XBTUSD', reverse=True, count=1).result()
pp.pprint(res)
res, http_response = bitMEXAuthenticated.Instrument.Instrument_get(symbol="XBTUSD").result()
pp.pprint(res)  # what I want is res['fundingRate'] (not 3x'd already) and res['fundingTimestamp'], res['indicativeFundingRate'] for next+8hrs pred

res, http_response = bitMEXAuthenticated.User.User_getMargin().result()
pp.pprint(res.amount) # access model fields directly, handy!
pp.pprint(res)
res, http_response = bitMEXAuthenticated.Execution.Execution_getTradeHistory(count=1).result()
pp.pprint(res)
res, http_response = bitMEXAuthenticated.OrderBook.OrderBook_getL2(symbol='XBTUSD', depth=1).result()
pp.pprint(res)
pp.pprint(res[0].side) # Multiple results are models in a list.



# Basic order placement
# print(dir(bitMEXAuthenticated.Order))
# res, http_response = bitMEXAuthenticated.Order.Order_new(symbol='XBTUSD', orderQty=3, price=1000).result()
# print(res)
