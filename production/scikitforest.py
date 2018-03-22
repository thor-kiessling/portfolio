import gc
import logging
import pickle
import sys
from datetime import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG,
    stream=sys.stdout)

from sknn import ae, mlp
from lasagne import layers as lasagne, nonlinearities as nl
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import globalvar

import os

os.nice(15)

errors = []
r2errors = []


def store_stats(avg_valid_error, avg_train_error, **_):
    errors.append((avg_valid_error, avg_train_error))


def extract_dyn_data():
    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []
    print("loading from DB")
    globalvar.dbcur.execute("SELECT * FROM calculated ORDER BY time ASC")
    data = globalvar.dbcur.fetchall()
    data = [i[:] for i in data if i[1] == 0]  # select only nonflip
    # dt = np.dtype(np.double)
    print("converting from DB")
    for index, line in enumerate(data):
        data[index] = np.frombuffer(line[2], dtype=np.double)
    # Iterate over the rows, splitting the label from the features.
    print("iterating over data")
    for index, line in enumerate(data):
        if index < len(data) - 5000: continue  # TODO temp - eliminates the first 100000
        averagenumerator = sum(float(x) for x in line[2:8])
        pricedelta = (averagenumerator / 6) - float(line[8])  # targeting sum { 1, 10, 20, 40, 80, 120

        #labels.append(float(pricedelta))
        if pricedelta >= 0:
            labels.append(1)
        else:
            labels.append(0)
        #       if pricedelta > 0.000550902932254: # this value gives 3 equal bins
        #           labels.append(2) # price will go up
        #       else:
        #           if pricedelta < -0.000550902932254:
        #               labels.append(0) #price will go down
        #           else:
        #               labels.append(1) #price will be neutral

        # fvecs.append([float(x) for x in line[8:20]])
        # fvecs.append([float(x) for x in line[81:]])
        fvecs.append([[float(x) for x in line[8:20]] + [float(y) for y in line[81:]]][0])
        if index % 1000 == 0:
            print(index / len(data))
    print("finalizing load")

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.matrix(fvecs).astype(np.float32)
    # Convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(np.float32)

    means = np.mean(fvecs_np, axis=0)
    fvecs_np -= means
    std_dev = np.std(fvecs_np, axis=0)
    # for item in std_dev:
    #     for subitem in item:
    #         for index, subsubitem in enumerate(subitem):
    #             if subitem == 0:
    #                 print(index, item, subitem, subsubitem, "std was zero")

    intermediate = (std_dev == 0).astype(int)
    std_dev = std_dev + intermediate
    fvecs_np /= std_dev
    std_dev_labels = np.std(labels_np, axis=0)

    # labels_np /= std_dev_labels
    # np.append(std_dev_labels, [[0], [0]])
    # print(std_dev_labels)

    np.savetxt("adjustments-mean.csv", means, delimiter=',')
    np.savetxt("adjustments-stddev.csv", std_dev, delimiter=',')
    # np.savetxt("adjustments-stddev-labels.csv", std_dev_labels, delimiter=',')




    # Convert the int numpy array into a one-hot matrix.
    labels_onehot = (np.arange(1) == labels_np[:, None]).astype(np.uint8)

    # Return a pair of the feature matrix and the one-hot label matrix.
    gc.collect()
    return fvecs_np, labels_np


def extract_plt_data(filename):
    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []

    # Iterate over the rows, splitting the label from the features.
    for line in open(filename):
        row = line.split(",")
        # p0 = float(row[0]) - float(row[3])  # set this and above to two for targeting 5.5m instad of 30
        p1 = float(row[1]) - float(row[3])
        p2 = float(row[2]) - float(row[3])
        labels.append(p2 + p1)
        #       if pricedelta > 0.000550902932254: # this value gives 3 equal bins
        #           labels.append(2) # price will go up
        #       else:
        #           if pricedelta < -0.000550902932254:
        #               labels.append(0) #price will go down
        #           else:
        #               labels.append(1) #price will be neutral

        fvecs.append([float(x) for x in row[3:] if x != 'nan'])

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.matrix(fvecs).astype(np.float32)



    # Convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(np.float32)

    # Convert the int numpy array into a one-hot matrix.
    # labels_onehot = (np.arange(2) == labels_np[:, None]).astype(np.uint8)

    # Return a pair of the feature matrix and the one-hot label matrix.

    return fvecs_np, labels_np


dyn = True

# X, Y = extract_data('train.csv')
# testX, testY = extract_data('test.csv')
if dyn:

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
    X = X.reshape(xshape[0], xshape[2])
    testX = testX.reshape(testxshape[0], testxshape[2])
    # Y = Y.reshape(yshape[0], 1)
    # testY = testY.reshape(testyshape[0], 1)
    print(X.shape, Y.shape, mainX.shape, mainY.shape, testX.shape, testY.shape)
    print(X.max, X.min, Y.max, Y.min)
    # Y = Y.reshape(yshape[0], yshape[2])
    # testY = testY.reshape(testyshape[0], testyshape[2])
    gc.collect()
    glob_rf = mlp.Regressor(
        layers=[
            mlp.Native(lasagne.DenseLayer, num_units=1024, nonlinearity=nl.very_leaky_rectify),
            mlp.Native(lasagne.DenseLayer, num_units=512, nonlinearity=nl.very_leaky_rectify),
            mlp.Native(lasagne.DenseLayer, num_units=256, nonlinearity=nl.very_leaky_rectify),
            mlp.Layer("Linear")],
        learning_rate=.1,
        n_iter=5,
        learning_rule="adadelta",
        callback={'on_epoch_finish': store_stats},

        loss_type='mse',
        regularize="L1",  # possibly L1, to instead filter out useless inputs. L1 gave 5+ in results?
        weight_decay=.001,  # default .0001 increase to combat overfitting.
        dropout_rate=0,  # keep 80% of neurons/inputs at .2, anti overfit
        verbose=True,
        #valid_set=(testX, testY),
        batch_size=1)  # TRIED NON-1, DIDN'T WORK AT ALL
    #glob_rf = pickle.load(open('forest' + str(length) + 'dyn.pkl', 'rb')) #TODO only for loading preexisting

    # begin pre-training with autoencoders
    try:
        glob_rf.fit(X, X)
    except KeyboardInterrupt:
        pass
    # move away from autoencoding to real training

    #print(glob_rf.get_parameters().shape()) #it's a list of tuples
    # (weight.np, bias.np, name str)
    weights = glob_rf.get_parameters()
    for index, layer in enumerate(weights):
        if layer[2] == "output":
            #print("deleting", weights[index])
            del weights[index]

    glob_rf = None

    glob_rf = mlp.Classifier(
        layers=[
            mlp.Native(lasagne.DenseLayer, num_units=1024, nonlinearity=nl.very_leaky_rectify),
            mlp.Native(lasagne.DenseLayer, num_units=512, nonlinearity=nl.very_leaky_rectify),
            mlp.Native(lasagne.DenseLayer, num_units=256, nonlinearity=nl.very_leaky_rectify),
            mlp.Layer("Softmax")],
        learning_rate=.005,
        n_iter=1,
        learning_rule="adadelta",
        callback={'on_epoch_finish': store_stats},

        loss_type='mcc',
        regularize="L1",  # possibly L1, to instead filter out useless inputs. L1 gave 5+ in results?
        weight_decay=.00001,  # default .0001 increase to combat overfitting.
        dropout_rate=0,  # keep 80% of neurons/inputs at .2, anti overfit
        verbose=True,
        valid_set=(testX, testY),
        batch_size=1)  # TRIED NON-1, DIDN'T WORK AT ALL TODO this block needs to be an exact clone of the one above
    glob_rf.set_parameters(weights)
    glob_rf.valid_set = (testX, testY)
    #glob_rf.hidden0 = None
    #glob_rf.hidden1 = None
    #glob_rf.hidden2 = None
    errors = []
    # )
    # gs = GridSearchCV(glob_rf, param_grid={
    #     'learning_rate': [0.05, 0.01, 0.005, 0.001],
    #     'weight_decay': [.1,.01,.001,.0001,.00001],
    #     'normalize': [None, 'batch'],
    #     'dropout_rate': [0, .1]
    #     })


    # glob_rf = joblib.load('forest' + str(length) + 'plt.pkl')
    print("starting training at", datetime.now())

    # note: current problem is ???

    # The bias is error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss
    #  the relevant relations between features and target outputs (underfitting).
    # The variance is error from sensitivity to small fluctuations in the training set. High variance can cause
    # overfitting: modeling the random noise in the training data, rather than the intended outputs.

    # takes 20min/cycle for (10,) 25 for (100,100,), 32 on 100x100x100
    # 45 mins for 300/250 etc. Consider removing the flipped ones for error.

    for i in range(500):
        # glob_rf.n_estimators += 10
        try:
            glob_rf.fit(X, Y)
        except KeyboardInterrupt:
            break
        # fitscore = glob_rf.score(X, Y)
        # r2errors.append([fitscore, glob_rf.score(testX, testY)])
        # print("Train R2", r2errors[-1][0], "Test R2", r2errors[-1][1],
        #       datetime.now())  # glob_rf.n_estimators, datetime.now())
        print("Train MCC:", errors[-1][1], "Test MCC", errors[-1][0], datetime.now())
        predicted = glob_rf.predict(testX)
        predicted = predicted.flatten()
        updownscore = 0
        for i in range(len(testY)):
            if int(testY[i]) == round(predicted[i]):
                updownscore += 1
        print("Test Accuracy:", updownscore / len(testY))
        # if np.std(predicted, axis=0) > .1:
        #     print("Convergence is good!", np.std(predicted, axis=0))
        # else:
        #     print("******ALERT CONVERGENCE IS BAD*******", np.std(predicted, axis=0))
        #print(gs.best_estimator_)
        # print(glob_rf.feature_importances_)
        # np.savetxt("featureimportances.csv", glob_rf.feature_importances_, delimiter=',')
        if len(errors) >= 20 and updownscore < oldfitscore: #errors[-1][1] >= errors[-2][1] + .1: #train stop if train no improve
            break
        else:
            pass
            #print(r2errors)
        pickle.dump(glob_rf, open('forest' + str(length) + 'dyn.pkl', 'wb'))
        oldfitscore = updownscore

    glob_rf = pickle.load(open('forest' + str(length) + 'dyn.pkl', 'rb'))
    # glob_rf = joblib.load('forest1276320dyn.pkl')
    predicted = glob_rf.predict(testX)
    predicted = predicted.flatten()
    trainpredict = glob_rf.predict(X)
    trainpredict = trainpredict.flatten()
    sns.set(style="darkgrid")
    print(errors)
    f1 = sns.jointplot(predicted, testY, color="r", kind="kde").plot_joint(
        sns.regplot, marker="+", scatter_kws={'alpha':0.3})  # , xlim=(-.02, .02), ylim=(-.02, .02), space=0)
    f1.set_axis_labels("predicted", "actual")  # x= predicted, always
    f1.fig.suptitle("Test data")

    f2 = sns.jointplot(trainpredict, Y, color="b", kind="kde").plot_joint(
        sns.regplot, marker="+", scatter_kws={'alpha':0.3})  # , xlim=(-.02, .02), ylim=(-.02, .02), space=0)
    f2.set_axis_labels("predicted", "actual")
    f2.fig.suptitle("Train data")

    f3 = plt.figure(2)
    plt.figure(2)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    errors = np.array(errors)
    y, x = errors[:, 1], errors[:, 0]
    t = np.arange(0, len(x), 1)
    ax1.plot(t, x, 'g-')
    ax1.set_ylabel('test', color='g')
    ax2.plot(t, y, 'b-')
    ax2.set_ylabel('train', color='b')

    updownscore = 0
    for i in range(len(testY)):
        if int(testY[i]) == round(predicted[i]):
            updownscore += 1
    print("Test Accuracy:", updownscore / len(testY))
    for i in range(len(Y)):
        if int(Y[i]) == round(trainpredict[i]):
            updownscore += 1
    print("Train Accuracy:", updownscore / len(Y))
    plt.show()
else:  # plot
    os.system("""awk 'seen[$0]++{print $0 > "dups.csv"; next}{print $0 > "dedupedata.csv"}' normalizeddata.csv""")
    os.system("rm dups.csv")
    mainX, mainY = extract_plt_data("dedupedata.csv")
    os.system("rm dedupedata.csv")
    X, Y, testX, testY = [], [], [], []
    length = len(mainX)
    X, testX, Y, testY = sklearn.model_selection.train_test_split(mainX, mainY, test_size=0.2, random_state=42)
    # for index, line in enumerate(mainX):
    #     if index < .75 * length:
    #         X.append(line)
    #     else:
    #         testX.append(line)
    #
    # for index, line in enumerate(mainY):
    #     if index < .75 * length:
    #         Y.append(line)
    #     else:
    #         testY.append(line)
    print()
    X = np.array(X)
    Y = np.array(Y)
    testX = np.array(testX)
    testY = np.array(testY)

    fitscore, oldfitscore = 0, 0
    print(X.shape, Y.shape, mainX.shape, mainY.shape, testX.shape, testY.shape)
    # xshape = X.shape
    # testxshape =testX.shape
    # yshape = Y.shape
    # testyshape = testY.shape
    # X = X.reshape(xshape[0],xshape[2])
    # testX = testX.reshape(testxshape[0], testxshape[2])
    print(X.shape, Y.shape, mainX.shape, mainY.shape, testX.shape, testY.shape)
    # Y = Y.reshape(yshape[0], yshape[2])
    # testY = testY.reshape(testyshape[0], testyshape[2])
    glob_rf = sklearn.ensemble.ExtraTreesRegressor(n_jobs=-1, n_estimators=0, warm_start=True)
    # glob_rf = joblib.load('forest' + str(length) + 'plt.pkl')
    print("starting training at", datetime.now())
    for i in range(50):
        glob_rf.n_estimators += 10
        glob_rf.fit(X, Y)
        fitscore = glob_rf.score(testX, testY)
        print(glob_rf.score(X, Y), fitscore, glob_rf.n_estimators, datetime.now())
        # print(glob_rf.feature_importances_)
        np.savetxt("featureimportances.csv", glob_rf.feature_importances_, delimiter=',')
        if fitscore < oldfitscore: break
        joblib.dump(glob_rf, 'forest' + str(length) + 'plt.pkl')
        oldfitscore = fitscore

    glob_rf = joblib.load('forest' + str(length) + 'plt.pkl')
    # glob_rf = joblib.load('forest904914plt.pkl')
    predicted = glob_rf.predict(testX)
    sns.set(style="darkgrid")

    f1 = sns.jointplot(predicted, testY, kind="reg", color="r", xlim=(-.02, .02), ylim=(-.02, .02),
                       space=0).plot_marginals(sns.distplot)
    updownscore = 0
    for i in range(len(testY)):
        if (testY[i] > 0) == (predicted[i] > 0):
            updownscore += 1
    print(updownscore / len(testY))
    plt.show()
