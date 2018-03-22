import gc
import logging
import pickle
import sys
from datetime import datetime
import math
from auto_ml import Predictor
import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.externals import joblib

#from gplearn.genetic import SymbolicRegressor

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


def extract_dyn_data():
    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []
    print("loading from DB")
    globalvar.dbcur.execute("SELECT * FROM calculated ORDER BY time ASC")#" LIMIT 1000000") #limit is to select only top x, put in place if swaps not mounted
    data = globalvar.dbcur.fetchall()
    #data = [i[:] for i in data if i[1] == 0]  # select only nonflip
    # dt = np.dtype(np.double)
    print("converting from DB")
    for index, line in enumerate(data):
        data[index] = np.frombuffer(line[2], dtype=np.double)
    # Iterate over the rows, splitting the label from the features.
    gc.collect()
    #data = [i.tolist() for i in data]
    print("iterating over data")
    for index, line in enumerate(data):
        #if index < len(data) - 200000: continue  # TODO this chooses the latest x datapoints
        averagenumerator = sum(float(x) for x in line[2:8])
        pricedelta = (averagenumerator / 6) - float(line[8])  # targeting sum { 1, 10, 20, 40, 80, 120
        # labels.append(float(pricedelta))
        # for item, line2 in enumerate(line):
        #     print(item, line2)
        # print(pricedelta)
        # break
        if pricedelta > 0.0: # this value gives 3 equal bins ~ish
            labels.append(1) # price will go up
        else:
            labels.append(0) #price will go down

        # fvecs.append([float(x) for x in line[8:20]])
        # fvecs.append([float(x) for x in line[81:]])
        temp_list = []
        temp_list = [float(x) for x in line[8:]]
        #print(len(temp_list))
        #print(len(fvecs))
        fvecs.append(temp_list)

        #print(fvecs)
        if index % 1000 == 0:
            print(index / len(data))
    print("finalizing load")

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.array(fvecs).astype(np.float32)
    # Convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(np.float32)

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

    # std_dev_labels = np.std(labels_np, axis=0)
    # labels_np /= std_dev_labels
    # np.append(std_dev_labels, [[0], [0]])
    # print(std_dev_labels)
    #
    # np.savetxt("adjustments-mean.csv", means, delimiter=',')
    # np.savetxt("adjustments-stddev.csv", std_dev, delimiter=',')
    # np.savetxt("adjustments-stddev-labels.csv", std_dev_labels, delimiter=',')



    # Convert the int numpy array into a one-hot matrix.
    # labels_onehot = (np.arange(2) == labels_np[:, None]).astype(np.uint8)

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


def getrandomsubslice(X_rand, testX_rand, size):
    randomfeatures = 0
    while randomfeatures == 0:
        randomfeatures = np.random.choice(X_rand.shape[1], size=size)
    randomfeatures = [0] + randomfeatures.tolist()
    X_slice = X_rand[:, randomfeatures]
    testX_slice = testX_rand[:, randomfeatures]
    return X_slice, testX_slice, randomfeatures


def getnonrandomslice(X_slice, testX_slice, features):
    X_slice = X_slice[:, features]
    testX_slice = testX_slice[:, features]
    return X_slice, testX_slice

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
    X = X.reshape(xshape[0], xshape[1])
    testX = testX.reshape(testxshape[0], testxshape[1])
    # Y = Y.reshape(yshape[0], 1)
    # testY = testY.reshape(testyshape[0], 1)
    print(X.shape, Y.shape, mainX.shape, mainY.shape, testX.shape, testY.shape)
    pd_train = pandas.DataFrame(X)
    pd_train['output'] = pandas.Series(Y, index=pd_train.index)
    pd_testX = pandas.DataFrame(testX)
    pd_testY = list(testY)
    print(pd_testX.shape)
    #print(X.max, X.min, Y.max, Y.min)
    # Y = Y.reshape(yshape[0], yshape[2])
    # testY = testY.reshape(testyshape[0], testyshape[2])
    gc.collect()
    # 3* Lasso
    # 2* Ridge



    #y_test_rdg = ridge.predict(testX)
    # )
    # gs = GridSearchCV(ridge, param_grid={
    #     'learning_rate': [0.05, 0.01, 0.005, 0.001],
    #     'hidden0__units': [10,100,1000,3000],
    #     'hidden1__units': [10,100,1000,3000],
    #     'weight_decay': [.1,.01,.001,.0001],
    #     'regularize': ["L1", "L2"]})
    #base_estimator = sklearn.linear_model.Ridge(alpha=.1)
    #ridge = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=50, learning_rate=.1)
    #ridge = sklearn.ensemble.RandomForestRegressor(bootstrap=False, n_estimators=400, max_depth=10,
    #                 n_jobs=10, warm_start=False, oob_score=False)#, min_impurity_split=.2) 50k pop = memoryerror
    # ga = SymbolicRegressor(population_size=10000, init_depth=(2, 10), parsimony_coefficient=0.0, tournament_size=20,
    #                        function_set=('add', 'sub', 'mul', 'div'),# 'sqrt', 'max', 'min', 'abs', 'sqrt', 'log', 'neg',
    #                                      #'inv', 'sin', 'cos', 'tan'),
    #                        p_crossover=0.7, p_subtree_mutation=0.1, max_samples=1.,
    #                        p_hoist_mutation=0.05, p_point_mutation=0.1,
    #                        verbose=1, n_jobs=12, generations=20, metric='mean absolute error')
    # ridgealpha = 300
    # print(ga)
    column_descriptions = {'output': 'output'}
    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    print("started training at", datetime.now())

    ml_predictor.train(pd_train, compute_power=5, X_test=pd_testX, y_test=pd_testY, perform_feature_scaling=True,
                       perform_feature_selection=False, optimize_entire_pipeline=True, optimize_final_model=False, #crashed on validation trying without op final model
                       take_log_of_y=False, model_names=[#'Ridge',
                       #'XGBRegressor', 'RANSACRegressor', 'RandomForestRegressor', 'AdaBoostRegressor',# 'LinearRegression',
                         "DeepLearningClassifier",]) # 'ExtraTreesRegressor', "LGBMRegressor", 'LogisticRegression',])
    ml_predictor.save()

# base estimator, train score, test score, test accu %  #TODO maybe add random noise to train data only
# BaggingRegressor,         .994, -4, .5284
# DecisionTreeRegressor,    .996, -.72, .53
# ExtraTreeRegressor,       .995, -1.43, .53
# ExtraTreesRegressor,      .998, -.98, .51
# GradientBoostingRegressor,.90,  -.17, .566 i liked this one, no multicore though
# HuberRegressor            .68,  -71,  .48
# KernelRidge total garbage
# LinearRegression total garbage
# LinearSVR                 .83, -108, .558
# NuSVR rbf, sigmoid, poly no work .808, -1.24, .566
# RANSACRegressor base Ridge .73, -9, .528 has potential
# RandomForestRegressor     .98, -1.5, .524 potential if able to regularize
# Ridge                     .76, -70, .59
# RidgeCV                   .87, -7000, .50
# SGDRegressor              negative a billion
# SVR                       .78, -.8, .522

    # ridge = joblib.load('forest' + str(length) + 'plt.pkl')

    # note: current problem is ???

    # The bias is error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss
    #  the relevant relations between features and target outputs (underfitting).
    # The variance is error from sensitivity to small fluctuations in the training set. High variance can cause
    # overfitting: modeling the random noise in the training data, rather than the intended outputs.

    # takes 20min/cycle for (10,) 25 for (100,100,), 32 on 100x100x100
    # 45 mins for 300/250 etc. Consider removing the flipped ones for error.
    tried_combos=[]
    bestfeature = [0, -1]
    list_good_feat = [[80, 291],[40, 269],[90, 303]]
    #for feature in itertools.permutations(range(X.shape[1]), r=2):
    for epoch in range(50):
        #if 0 in feature: continue
        #feature = list(feature)
        #if feature in tried_combos: continue
        #tried_combos.append([feature[1], feature[0]])
        #feature = [0] + feature
        #X_slice, testX_slice = getnonrandomslice(X, testX, feature)

        try:
            # XD = np.random.rand(X_slice.shape[0], X_slice.shape[1] )
            # XD -= .5
            # XD *= 2
            # XD += X_slice
            ga.fit(X, Y)
            print(ga._program)
            fitscore = ga.score(X, Y)
            r2errors.append([fitscore, ga.score(testX, testY)])
            #print("using n estimators:", ridgealpha)
            #print("OOB score:", ridge.oob_score_)
            print("Train R2", r2errors[-1][0], "Test R2", r2errors[-1][1],
                  datetime.now())  # ridge.n_estimators, datetime.now())
            #print("Train MSE:", errors[-1][1], "Test MSE", errors[-1][0], datetime.now())
            predicted = ga.predict(testX)
            predicted = predicted.flatten()
            # updownscore = 0
            # for i in range(len(testY)):
            #     if (testY[i] > 0) == (predicted[i] > 0):
            #         updownscore += 1
            # print("Test Accuracy:", updownscore / len(testY))
            if np.std(predicted, axis=0) > .1:
                 print("Convergence is good!", np.std(predicted, axis=0), np.std(testY, axis=0))
            else:
                 print("******ALERT CONVERGENCE IS BAD*******", np.std(predicted, axis=0), np.std(testY, axis=0))
            if len(r2errors) > 2 and r2errors[-1][1] < r2errors[-2][1]:
                #if bestfeature[0] != 0:
                #    os.remove("rf" + str(bestfeature[0]) + ".pkl")
                #bestfeature[1] = r2errors[-1][1]  # the testX score
                #bestfeature[0] = feature          # which features

                break
            if r2errors[-1][1] >= max(b for (a, b) in r2errors):
                print("saved model")
                pickle.dump(ga, open('ga' + str(len(X)) + '.pkl', 'wb'))
        except KeyboardInterrupt:
            print("Received KB interrupt, loading model and evaluating")
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
    #X_slice, testX_slice = getnonrandomslice(X, testX, bestfeature[0])
    ridge = pickle.load(open('ga' + str(len(X)) + '.pkl', 'rb'))
    print("Loaded best model, running diagnostics")
    print(ridge._program)
    fitscore = ridge.score(X, Y)
    r2errors.append([fitscore, ridge.score(testX, testY)])
    # print("using n estimators:", ridgealpha)
    # print("OOB score:", ridge.oob_score_)
    print("Train R2", r2errors[-1][0], "Test R2", r2errors[-1][1],
          datetime.now())
    # ridge = joblib.load('forest1276320dyn.pkl')
    predicted = ridge.predict(testX)
    predicted = predicted.flatten()
    trainpredict = ridge.predict(X)
    trainpredict = trainpredict.flatten()
    if np.std(predicted, axis=0) > .1:
        print("Convergence is good!", np.std(predicted, axis=0), np.std(testY, axis=0))
    else:
        print("******ALERT CONVERGENCE IS BAD*******", np.std(predicted, axis=0), np.std(testY, axis=0))
    sns.set(style="darkgrid")

    f1 = sns.jointplot(predicted, testY, color="r", kind="kde").plot_joint(
        sns.regplot, marker="+", scatter_kws={'alpha': 0.3})  # , xlim=(-.02, .02), ylim=(-.02, .02), space=0)
    f1.set_axis_labels("predicted", "actual")  # x= predicted, always
    f1.fig.suptitle("Test data")

    f2 = sns.jointplot(trainpredict, Y, color="b", kind="kde").plot_joint(
        sns.regplot, marker="+", scatter_kws={'alpha': 0.1})  # , xlim=(-.02, .02), ylim=(-.02, .02), space=0)
    f2.set_axis_labels("predicted", "actual")
    f2.fig.suptitle("Train data")

    f3 = plt.figure(3)
    plt.figure(3)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    errors = np.array(r2errors)
    y, x = errors[:, 0], errors[:, 1]
    t = np.arange(0, len(x), 1)
    ax1.plot(t, x, 'g-')
    ax1.set_ylabel('test', color='g')
    ax2.plot(t, y, 'b-')
    ax2.set_ylabel('train', color='b')

    updownscore = 0
    for i in range(len(testY)):
        if (testY[i] > 0) == (predicted[i] > 0):
            updownscore += 1
    print("Test accu:", updownscore / len(testY))
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
    ridge = sklearn.ensemble.ExtraTreesRegressor(n_jobs=-1, n_estimators=0, warm_start=True)
    # ridge = joblib.load('forest' + str(length) + 'plt.pkl')
    print("starting training at", datetime.now())
    for i in range(50):
        ridge.n_estimators += 10
        ridge.fit(X, Y)
        fitscore = ridge.score(testX, testY)
        print(ridge.score(X, Y), fitscore, ridge.n_estimators, datetime.now())
        # print(ridge.feature_importances_)
        np.savetxt("featureimportances.csv", ridge.feature_importances_, delimiter=',')
        if fitscore < oldfitscore: break
        joblib.dump(ridge, 'forest' + str(length) + 'plt.pkl')
        oldfitscore = fitscore

    ridge = joblib.load('forest' + str(length) + 'plt.pkl')
    # ridge = joblib.load('forest904914plt.pkl')
    predicted = ridge.predict(testX)
    sns.set(style="darkgrid")

    f1 = sns.jointplot(predicted, testY, kind="reg", color="r", xlim=(-.02, .02), ylim=(-.02, .02),
                       space=0).plot_marginals(sns.distplot)
    updownscore = 0
    for i in range(len(testY)):
        if (testY[i] > 0) == (predicted[i] > 0):
            updownscore += 1
    print(updownscore / len(testY))
    plt.show()
