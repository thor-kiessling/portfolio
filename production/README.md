# Portfolio

Hello, I am Thor Kiessling and this is my showcase for real-world python implementations of machine learning and 
algorithmic trading. To get these scripts running rename api_keys_example.py to api_keys.py and input
 [bitmex.com](https://www.bitmex.com) api keys. If you wish to recompile the Cython portions:
 ```
     python setup.py build_ext --inplace 
 ```


## Employers

[Bitmex sniper](production/bitmex_sniper.py) is a multi-threaded high performance python algorithmic trading script. It features
switching between long term and short term business logic, C++ comparable execution time on latency-sensitive tasks, and 
integration with compiled Cython modules.

My primary example of network optimization is the [websocket thread](production/market_maker/ws/ws_thread.py) module 
that handles real-time concurrent state synchronization, serving 
data under load of bitmex's Level 2 orderbook (10-100 diffs/second) on the resources of an AWS t2.micro instance. It 
manages the exchange, orderbook, user balance, and user position state for the _BitMEX_ class.

[My fork](production/motif_matching_knn.pyx) of [nicholasg3's motif matching](https://github.com/nicholasg3/motif-mining/blob/master/Motif_Matching.py)
extends the original by adding options for K-nearest neighbor matching and vastly speeds up execution time with Cython
cdefs and compiling. Runtime improved from 41 seconds to .26 seconds when matching a 100 length sample to a 568k length 
timeseries. The [optimization report](production/motif_matching_knn.html) shows how this was achieved, with the white sections of 
code inside loops converting to C without python protection/interaction.

Data visualization is essential to understanding what my code is doing. Scatter plots and histograms helped me to
pinpoint trends, bugs, and other problems. Straightforward price and performance stats helped me decide which 
strategies to pursue or abandon. These images were generated by [ucr classifier](production/ucr_classifier.py), 
a script to train the hyperparameters of a knn motif matching trading strategy.

![alt text](https://i.imgur.com/8MiND4S.png) ![alt text](https://i.imgur.com/eJDtDl4.png) 



## Machine Learning 

[Wavenet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) is a Google Deepmind project to iteratively 
generate realistic voice with a deep convolutional neural network. I adapted the technique to arbitrary time series and
attempted to use the wave generator to predict price movements. [Generator](production/wavenet_generate.py),
[trainer](production/wavenet_trainer.py), and [evaluator as a trade strategy](production/wavenet-eval.py). I was 
unsuccessful in producing a viable trading strategy but learned a lot about [tensorflow](https://www.tensorflow.org/) internals.

Feature production and selection is essential in any data analysis. Feeding raw (or normalized) time series into almost every 
machine learning technique will result in failure, the time series must be transformed into actionable features. My 
module for doing this is based on [ta-lib](https://github.com/mrjbq7/ta-lib) and is called [build_ta_data](production/build_ta_data.py)
It was able to grab random sections of the time series, run a full TA suite on that subsequence, and store the results. 
Later experiments included using an [autoencoder](production/scikitforest.py) to determine feature importances.


## Algotrading

[DB.sqlite](production/DB.sqlite) contains all 1m OHLC data for XBTUSD in _bmxswap_, all funding history for XBTUSD in 
_bmxfund_, and an incomplete 1m OHLC for okcoin quarterlies.

Bitmex sniper incorporates all I have learned about the mechanics of algotrading, it reliably executes long term and 
short term positions with variable leverage options. I hope someone out there finds it useful to copy and modify or simply 
learn from. It was forked from bitmex's [sample market maker](https://github.com/BitMEX/sample-market-maker). any files 
with _run.py are meant to be executed in production instead of the main file, providing restart on looping and Cython 
performance improvements.

The [runlive](production/runlive.py) script is designed to work with OKEX and does one of the best jobs I've seen of implementing
their API. I advise against trading there as there are better futures options elsewhere and they frequently change their API without
any documentation.