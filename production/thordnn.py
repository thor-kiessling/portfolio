# -*- coding: utf-8 -*-

""" Deep Neural Network for MNIST dataset classification task.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
import os
import math
import tensorflow as tf
import seaborn as sns
import matplotlib as plt
import gc


def extract_data(filename):
    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []

    # Iterate over the rows, splitting the label from the features. Convert labels
    # to integers and features to floats.
    print("starting to load")
    zerobound = .00066  # should result in 4 equal bins
    for line in open(filename):
        row = line.split(",")
        label = float(row[1]) - float(row[3])  # targeting the 10m price with row[1]. gives delta, general range +-.001
        # p1 = float(row[1]) - float(row[3])
        # p2 = float(row[2]) - float(row[3])
        if label < 0:
            if label < -zerobound:
                label_mod = 0  # "bigdown"
            else:
                label_mod = 1  # "down"
        else:
            if label > zerobound:
                label_mod = 3  # "bigup"
            else:
                label_mod = 2  # "up"
        labels.append(label_mod)
        fvecs.append([float(x) for x in row[3:] if x != 'nan'])

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.array(fvecs).astype(np.float32)
    fvecs_np = fvecs_np[:, [254,106,125,195,205,235,196,185,170,255,219,107,174,159,145,294,300,285,165,204,310,220,206,295,180,101,284,146,240,245,299,171,276,160,186,175,102,124,150,271,121,261,169,164,309,234,179,236,184,140,126,250,116,82,256,2,239,244,97,105,141,241,325,86,311,260,301,324,246,96,221,194,81,120,166,151,77,181,321,341,131,87,176,161,326,251,144,266]]
    # Convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(dtype=np.uint8)

    #compute the mean and std-dev normalization, save them, apply them
    means = np.mean(fvecs_np, axis=0)
    fvecs_np -= means
    std_dev = np.std(fvecs_np, axis=0)
    fvecs_np /= std_dev
    np.savetxt("adjustments-mean.csv", means, delimiter=',')
    np.savetxt("adjustments-stddev.csv", std_dev, delimiter=',')

    print("loaded.")


    # Convert the array of int labels into a numpy array.
    #labels_np = np.array(labels).astype(np.float32)

    # Convert the int numpy array into a one-hot matrix.
    labels_onehot = (np.arange(4) == labels_np[:, None]).astype(np.float32)

    # Return a pair of the feature matrix and the one-hot label matrix.

    return fvecs_np, labels_onehot


# Data loading and preprocessing
# import tflearn.datasets.mnist as mnist
# X, Y, testX, testY = mnist.load_data(one_hot=True)
# import tflearn.data_utils as data_utils
# X, Y = data_utils.load_csv("train.csv", target_column=2, columns_to_ignore=(0,1), has_header=False, categorical_labels=False)
# testX, testY = data_utils.load_csv("test.csv", target_column=2, columns_to_ignore=(0,1), has_header=False, categorical_labels=False)
activation = "relu6"
os.system("""awk 'seen[$0]++{print $0 > "dups.csv"; next}{print $0 > "dedupedata.csv"}' normalizeddata.csv""")
os.system("rm dups.csv")
mainX, mainY = extract_data("dedupedata.csv")
os.system("rm dedupedata.csv")
X, Y, testX, testY = [], [], [], []
length = len(mainX)
for index, line in enumerate(mainX):
    if index < .8 * length:
        X.append(line)
    else:
        testX.append(line)

for index, line in enumerate(mainY):
    if index < .8 * length:
        Y.append(line)
    else:
        testY.append(line)
print()
X = np.array(X)
Y = np.array(Y)
testX = np.array(testX)
testY = np.array(testY)
print(X.shape, Y.shape, mainX.shape, mainY.shape, testX.shape, testY.shape)
xshape = X.shape
testxshape = testX.shape
#X = X.reshape(xshape[0], xshape[2])
#testX = testX.reshape(testxshape[0], testxshape[2])
print(X.shape, Y.shape, mainX.shape, mainY.shape, testX.shape, testY.shape)
def do_dnn(): #optimizer, loss, activation):
    # print(y1.shape, testy1.shape)
    #data_prep = tflearn.data_preprocessing.DataPreprocessing()
    #data_prep.add_samplewise_zero_center()
    #data_prep.add_samplewise_stdnorm()

    # Building deep neural network
    xshape = X.shape
    activation = "relu6"
    input_layer = tflearn.input_data(shape=[None, xshape[1]])# , data_preprocessing=data_prep)
    dense1 = tflearn.fully_connected(input_layer, 40, activation=activation,
                                     regularizer='L2', weight_decay=0.001)

    # install a deep network of highway layers
    highway = dense1
    for i in range(1):
        highway = tflearn.highway(highway, 40, activation=activation,
                                  regularizer='L2', weight_decay=0.001, transform_dropout=0.726)

    softmax = tflearn.fully_connected(highway, 4, activation="softmax")
    #final = tflearn.layers.merge_outputs(highway)

    # Regression using SGD with learning rate decay and accuracy
    # sgd = tflearn.SGD(learning_rate=.001, lr_decay=0.96, decay_step=1000)
    # top_k = tflearn.metrics.Top_k(1)
    # lossfunc = tflearn.objectives.softmax_categorical_crossentropy(softmax, )
    #adam = tflearn.optimizers.Adam(learning_rate=.001, beta1=.9, beta2=.99, epsilon=math.pow(10, -8))
    # .001 is default, radically raised? Might be wrong. .1 and .00001 didn't work with 10k batches
    adagrad = tflearn.optimizers.AdaGrad(learning_rate=.036)
    # must use batchsize = 1
    top3 = tflearn.metrics.Top_k(3)

    net = tflearn.regression(softmax, optimizer=adagrad, metric="accuracy", loss="mean_square")# , #n_classes=11,
                            # to_one_hot=True, batch_size=100000) # metric=normal_R2 #categorical_crossentropy
    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path='/home/thorbinator/PycharmProjects/production/saves/save',
                        best_checkpoint_path='/home/thorbinator/PycharmProjects/production/saves/best',
                        max_checkpoints=4, best_val_accuracy=.31)
    # model.load('./saves/save-30000')
    #print (final.get_shape().as_list() , y1.get_shape().as_list(), X.get_shape().as_list())
    model.fit(X, Y, validation_set=(testX, testY), n_epoch=99999, show_metric=True, run_id="th_run",
              snapshot_epoch=True, batch_size=1, snapshot_step=75000)

def main():
    do_dnn()
    # modelevaluation()
    # tf.reset_default_graph()
def modelevaluation():
    input_layer = tflearn.input_data(shape=[None, 392])  # , data_preprocessing=data_prep)
    dense1 = tflearn.fully_connected(input_layer, 100, activation=activation,
                                     regularizer='L2', weight_decay=0.001)

    # install a deep network of highway layers
    highway = dense1
    for i in range(100):
        highway = tflearn.highway(highway, 100, activation=activation,
                                  regularizer='L2', weight_decay=0.001, transform_dropout=0.8)

    softmax = tflearn.fully_connected(highway, 10, activation="softmax")
    # final = tflearn.layers.merge_outputs(highway)

    # Regression using SGD with learning rate decay and accuracy
    # sgd = tflearn.SGD(learning_rate=.001, lr_decay=0.96, decay_step=1000)
    # top_k = tflearn.metrics.Top_k(1)
    # lossfunc = tflearn.objectives.softmax_categorical_crossentropy(softmax, )
    adam = tflearn.optimizers.Adam(learning_rate=.00001, beta1=.9, beta2=.99, epsilon=math.pow(10, -8))
    # .001 is default, radically raised? Might be wrong. .1 and .00001 didn't work with 10k batches
    top3 = tflearn.metrics.Top_k(3)

    net = tflearn.regression(softmax, optimizer=adam, metric='accuracy', loss="categorical_crossentropy")  # , #n_classes=11,
    # to_one_hot=True, batch_size=100000) # metric=normal_R2
    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0,
                        # checkpoint_path='/home/thorbinator/PycharmProjects/production/saves/save',
                        best_checkpoint_path='/home/thorbinator/PycharmProjects/production/saves/best',
                        max_checkpoints=4, best_val_accuracy=.31)
    model.load('./saves/save-300000')
    minitestY, minitestX = [], []
    for index, value in enumerate(testY):
        if index < 100: minitestY.append(value)
    for index, value in enumerate(testX):
        if index < 100: minitestX.append(value)
    minitestY, minitestX = np.array(minitestY), np.array(minitestX)
    results = model.evaluate(minitestX, minitestY,
                                 batch_size=1)
    for key in sorted(results):
        print(key)
    updownscore = 0
    score4bin = 0
    bad4bin = 0
    offbyone = 0
    predicted = []
    offbytwo = 0
    zerobound = .0001  # 0.000550902932254
    for index, result in enumerate(model.predict(minitestX)):
        if label < 0:
            if label < -zerobound:
                label_mod = 0  # "bigdown"
            else:
                label_mod = 1  # "down"
        else:
            if label > zerobound:
                label_mod = 3  # "bigup"
            else:
                label_mod = 2  # "up"
        if result < 0:
            if result < -zerobound:
                pred4bin = 0  # "bigdown"
            else:
                pred4bin = 1  # "down"
        else:
            if result > zerobound:
                pred4bin = 3  # "bigup"
            else:
                pred4bin = 2  # "up"

        if pred4bin == label_mod:
            score4bin += 1
        if abs(pred4bin - label_mod) == 3:
            bad4bin += 1
        if abs(pred4bin - label_mod) == 2:
            offbytwo += 1
        if abs(pred4bin - label_mod) == 1:
            offbyone += 1

        if (label > 0) == (result > 0):
            updownscore += 1
        if index % 100 == 0: print(label, ":", result, ":", label_mod, ":", pred4bin)
        predicted.extend(result)
    print("updown correctness:", updownscore / len(testY))
    print("4 bins correctness:", score4bin / len(testY))
    print("off by one category:", offbyone / len(testY))
    print("off by two categories:", offbytwo / len(testY))
    print("really bad predictions:", bad4bin / len(testY))
    sns.set(style="darkgrid")
    sns.jointplot(np.array(predicted), testY, kind="reg", color="r", xlim=(-.01, .01), ylim=(-.01, .01), size=40,
                  space=0).plot_marginals(sns.distplot, bins=40)
    plt.show()

if __name__ == "__main__":
    main()



