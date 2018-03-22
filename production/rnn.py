import gc
import logging
import pickle
import sys
from datetime import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools

logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG,
    stream=sys.stdout)

#from sknn import ae, mlp
#from lasagne import layers as lasagne, nonlinearities as nl
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import globalvar

import os

os.nice(15)

errors = []
r2errors = []


def store_stats(avg_valid_error, avg_train_error, **_):
    errors.append((avg_valid_error, avg_train_error))

num_samples = 1000
look_back_window = 100
minutes_look_ahead = 0

def extract_dyn_data():
    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []
    print("loading from DB")
    globalvar.dbcur.execute("SELECT * FROM futures ORDER BY time ASC")
    data = globalvar.dbcur.fetchall()
    sequence_length = look_back_window

    # dt = np.dtype(np.double)
    # print("converting from DB")
    # for index, line in enumerate(data):
    #     data[index] = np.frombuffer(line[2], dtype=np.double)
    # Iterate over the rows, splitting the label from the features.
    print("iterating over data")
    for index, line in enumerate(data):
        temp_list = []
        if index < len(data) - num_samples - minutes_look_ahead: continue  # TODO temp - eliminates the first 100000
        if index > len(data) - minutes_look_ahead - 1: continue
        #averagenumerator = sum(float(x) for x in line[2:8])
        #pricedelta = (averagenumerator / 6) - float(line[8])  # targeting sum { 1, 10, 20, 40, 80, 120


        #print(data[index][4])
        #print(data[index-1][4])
        for i in range(sequence_length):
            temp_list.append([float(x) for x in data[index-sequence_length + i][1:]])  # properly grabs up to index-1
        # mean = np.array(temp_list).mean(axis=0)
        # mean = mean[3]
        # temp_list = np.array(temp_list)
        # temp_list /= mean
        # temp_list = temp_list.tolist()
        #print (mean, temp_list, data[index][4] / mean)  # all of this is sucessful batchwise normalizing
        fvecs.append(temp_list)
        temp_list = [float(x) for x in data[index+minutes_look_ahead][1:]]
        # temp_list = np.array(temp_list)
        # temp_list /= mean
        # temp_list = temp_list.tolist()
        labels.append(temp_list)  # predict the next minute close
        #print(temp_list[-1][3])
        #print(np.array(fvecs).shape)
        #print(fvecs[-1])  #todo debug only
        if index % 1000 == 0:
            print(index / len(data))
    print("finalizing load")

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.array(fvecs).astype(np.float32)
    labels_np = np.array(labels).astype(np.float32)
    print(np.mean(labels_np, axis=0))
    means = np.mean(labels_np, axis=0)
    labels_np -= means
    fvecs_np -= means
    print(np.std(labels_np, axis=0))
    fvecs_np /= np.std(labels_np, axis=0)
    labels_np /= np.std(labels_np, axis=0)

    # means = np.mean(fvecs_np, axis=0)
    # fvecs_np -= means
    # std_dev = np.std(fvecs_np, axis=0)
    # for item in std_dev:
    #     for subitem in item:
    #         for index, subsubitem in enumerate(subitem):
    #             if subitem == 0:
    #                 print(index, item, subitem, subsubitem, "std was zero")

    # intermediate = (std_dev == 0).astype(int)
    # std_dev = std_dev + intermediate
    # fvecs_np /= std_dev



    # np.savetxt("adjustments-mean.csv", means, delimiter=',')
    # np.savetxt("adjustments-stddev.csv", std_dev, delimiter=',')

    # Convert the array of int labels into a numpy array.


    # Return a pair of the feature matrix and the one-hot label matrix.
    gc.collect()
    return fvecs_np, labels_np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras.regularizers import l1l2, activity_l2
from keras.layers.advanced_activations import LeakyReLU



# X, Y = extract_data('train.csv')
# testX, testY = extract_data('test.csv')

mainX, mainY = extract_dyn_data()
X, Y, testX, testY = [], [], [], []
length = len(mainX)
# X, testX, Y, testY = sklearn.model_selection.train_test_split(mainX, mainY, test_size=0.2, random_state=42)
for index, line in enumerate(mainX):
    if index < .90 * length:
        X.append(line)
    else:
        testX.append(line)

for index, line in enumerate(mainY):
    if index < .90 * length:
        Y.append(line)
    else:
        testY.append(line)
print()
gc.collect()
X = np.array(X)
Y = np.array(Y)
testX = np.array(testX)
testY = np.array(testY)



fitscore, oldfitscore = 0, 100000000
print(X.shape, Y.shape, mainX.shape, mainY.shape, testX.shape, testY.shape)
xshape = X.shape
testxshape = testX.shape
yshape = Y.shape
testyshape = testY.shape
#X = X.reshape(xshape[0], xshape[2])
#testX = testX.reshape(testxshape[0], testxshape[2])
# Y = Y.reshape(yshape[0], 1)
# testY = testY.reshape(testyshape[0], 1)
print(X.shape, Y.shape, mainX.shape, mainY.shape, testX.shape, testY.shape)
#print(X.max, X.min, Y.max, Y.min)
# Y = Y.reshape(yshape[0], yshape[2])
# testY = testY.reshape(testyshape[0], testyshape[2])
batch_size = 1
in_out_neurons = 5
hidden_neurons = 500
print("started training at", datetime.now())
model = Sequential()
model.add(LSTM(hidden_neurons, batch_input_shape=(batch_size, look_back_window, 5), return_sequences=False, stateful=True, W_regularizer=l1l2(.0001, .0001), U_regularizer=l1l2(.0001, .0001)))
model.add(Activation('linear'))
model.add(Dense(in_out_neurons))
model.compile(loss="mse", optimizer="nadam", metrics=['mean_squared_logarithmic_error'])

# note: current problem is ???

# The bias is error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss
#  the relevant relations between features and target outputs (underfitting).
# The variance is error from sensitivity to small fluctuations in the training set. High variance can cause
# overfitting: modeling the random noise in the training data, rather than the intended outputs.

# takes 20min/cycle for (10,) 25 for (100,100,), 32 on 100x100x100
# 45 mins for 300/250 etc. Consider removing

for epoch in range(50):
    try:


        #for j in range(look_back_window):
        #print(X[i].shape)
        #print(np.expand_dims(X[i], axis=0).shape, np.array([y_true]).shape)
        model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=1, shuffle=False)
        te_loss, te_acc = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
        print('rmsle testing = {}'.format(np.mean(te_acc)))
        print('loss testing = {}'.format(np.mean(te_loss)))
        print('___________________________________')
        r2errors.append(te_acc)
        model.reset_states()
        x_pred = model.predict(X, batch_size=batch_size)
        y_pred = model.predict(testX, batch_size=batch_size)

        model.reset_states()

        print('rmsle testing = {}'.format(np.mean(te_acc)))
        print('loss testing = {}'.format(np.mean(te_loss)))
        print('___________________________________')
        #fitscore = model.evaluate(X, Y, batch_size=1)
        #print("using n estimators:", ridgealpha)
        #print("OOB score:", ridge.oob_score_)
        print()
        print("Test rmsle", r2errors[-1],
              datetime.now())  # ridge.n_estimators, datetime.now())
        #print("Train MSE:", errors[-1][1], "Test MSE", errors[-1][0], datetime.now())
        predicted = np.array(y_pred)
        predicted = predicted.reshape(testY.shape)
        testpredict = np.array(x_pred)
        testpredict = testpredict.reshape(Y.shape)
        testY_col = testY[:, [3]]
        Y_col = Y[:, [3]]
        predicted = predicted[:, [3]]
        testpredict = testpredict[:, [3]]
        #print(predicted, testY_col) #todo debug
        updownscore = 0
        for i in range(len(testY)):
            if (testY_col[i] > testX[i][-1][3]) == (predicted[i] > testX[i][-1][3]):
                updownscore += 1
        print("Test Accuracy:", updownscore / len(testY))
        updownscore = 0
        for i in range(len(Y)):
            if (Y_col[i] > X[i][-1][3]) == (testpredict[i] > X[i][-1][3]):
                updownscore += 1
        print("Train Accuracy:", updownscore / len(Y))
        if np.std(predicted, axis=0) > .5*np.std(testY_col, axis=0) and np.std(predicted, axis=0) < 2*np.std(testY_col, axis=0) :
            print("Convergence is good!", np.std(predicted, axis=0), np.std(testY_col, axis=0))
        else:
            print("******ALERT CONVERGENCE IS BAD*******", np.std(predicted, axis=0), np.std(testY_col, axis=0))
        if len(r2errors) >= 26 and r2errors[-1] > r2errors[-2]:
            break
        model.save('rnn' + str(len(mainX)) + '.h5')
        model.reset_states()
    except KeyboardInterrupt:
        print("Received KB interrupt, loading model and evaluating")
        model.reset_states()
        break
    # print(gs.best_estimator_)
    # print(ridge.feature_importances_)
    # np.savetxt("featureimportances.csv", ridge.feature_importances_, delimiter=',')
    # if len(r2errors) >= 35 and r2errors[-1][1] <= r2errors[-2][1]: #train stop if test no improve
    #     break


    # ridgealpha += 10
    #
    # ridge.n_estimators = ridgealpha
    # oldfitscore = errors[-1][1]

model = load_model('rnn' + str(len(mainX)) + '.h5')
print("Loaded best model, running diagnostics")

# print("using n estimators:", ridgealpha)
# print("OOB score:", ridge.oob_score_)
print("Train R2", r2errors[-1][0], "Test R2", r2errors[-1][1],
      datetime.now())
# ridge = joblib.load('forest1276320dyn.pkl')
predicted = model.predict(testX, batch_size=1)
trainpredict = model.predict(X, batch_size=1)
predicted = predicted.reshape(testY.shape)
trainpredict = trainpredict.reshape(Y.shape)
testY = testY[:, [3]]
predicted = predicted[:, [3]]
trainpredict = trainpredict[:, [3]]
Y = Y[:, [3]]
updownscore =0
for i in range(len(testY)):
    if (testY[i] > testX[i][-1][3]) == (predicted[i] > testX[i][-1][3]):
        updownscore += 1
print("Test Accuracy:", updownscore / len(testY))
updownscore = 0
for i in range(len(Y)):
    if (Y[i] > X[i][-1][3]) == (trainpredict[i] > X[i][-1][3]):
        updownscore += 1
print("Train Accuracy:", updownscore / len(Y))
if np.std(predicted, axis=0) > .5 * np.std(testY, axis=0) and np.std(predicted, axis=0) < 2 * np.std(testY, axis=0):
    print("Convergence is good!", np.std(predicted, axis=0), np.std(testY, axis=0))
else:
    print("******ALERT CONVERGENCE IS BAD*******", np.std(predicted, axis=0), np.std(testY, axis=0))
sns.set(style="darkgrid")
# for i, line in enumerate(testY):
#     print(testX[i][-1][3], "fed into v")
#     print(line, "<- actual| pred ->", predicted[i])
#print(predicted, testY)
#print(trainpredict, Y)
f1 = sns.jointplot(predicted, testY, color="r", kind="kde").plot_joint(
    sns.regplot, marker="+", scatter_kws={'alpha': 0.3})  # , xlim=(-.02, .02), ylim=(-.02, .02), space=0)
f1.set_axis_labels("predicted", "actual")  # x= predicted, always
f1.fig.suptitle("Test data")

f2 = sns.jointplot(trainpredict, Y, color="b", kind="kde").plot_joint(
    sns.regplot, marker="+", scatter_kws={'alpha': 0.1})  # , xlim=(-.02, .02), ylim=(-.02, .02), space=0)
f2.set_axis_labels("predicted", "actual")
f2.fig.suptitle("Train data")

f3 = plt.figure(2)
plt.figure(2)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
errors = np.array(r2errors)
y, x = errors[:, 0], errors[:, 1]
t = np.arange(0, len(x), 1)
ax1.plot(t, x, 'g-')
ax1.set_ylabel('test', color='g')
ax2.plot(t, y, 'b-')
ax2.set_ylabel('train', color='b')

plt.show()