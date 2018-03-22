# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A stand-alone example for tf.learn's random forest model on mnist."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest

os.nice(19)
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', './thorforest/', 'Base directory for output models.')
flags.DEFINE_string('data_dir', './saves/', 'Directory for storing data')

flags.DEFINE_integer('train_steps', 1000, 'Number of training steps.')
flags.DEFINE_string('batch_size', 1000,
                    'Number of examples in a training batch.')
flags.DEFINE_integer('num_trees', 1200, 'Number of trees in the forest.')
flags.DEFINE_integer('max_nodes', 50000, 'Max total nodes in a single tree.')


# 300,10000 got .71 accuracy
# 10,100 got .55 acc
# 600,10000 used 4gb ram
# 600,100000 used 14gb, .725 acc, 21201 steps 17 hours
# 1200,10000 used 11 gig, .715 acc, 2601 steps
# 2000,10000 very very slow, .2 steps/sec
# 300,10000 with new triple bin .636, 2701 steps
# 1200,10000  w/reg 5001 steps .2688 R2, .667 binary acc. .719 4bins acc, .282 reallybad 4 hours to run
# 1200,50000 w/reg 28602 steps, .276 R2, .69 binary acc, .731 4bins, .268 wrong, .6pearsonr
# 1200,50000 now aimed at 1m+10m avg

def extract_data(filename):
    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []

    # Iterate over the rows, splitting the label from the features. Convert labels
    # to integers and features to floats.
    for line in open(filename):
        row = line.split(",")
        averagenumerator = sum(float(x) for x in row[0:2])
        pricedelta = (averagenumerator / 2) - float(row[3])
        labels.append(float(pricedelta))
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

    # check for pesky nans
    for index, row in enumerate(fvecs):
        for index2, item in enumerate(row):
            if np.isnan(item):
                print("nan found")
                print(index2, item)
                print(index, row)
                return 1

    # Convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(np.float32)

    # Convert the int numpy array into a one-hot matrix.
    # labels_onehot = (np.arange(2) == labels_np[:, None]).astype(np.uint8)

    # Return a pair of the feature matrix and the one-hot label matrix.

    return fvecs_np, labels_np  # labels_onehot #removed onehot changeover


def build_estimator(model_dir):
    """Build an estimator."""
    params = tensor_forest.ForestHParams(
        num_classes=1, regression=True, num_features=392,
        num_trees=FLAGS.num_trees, max_nodes=FLAGS.max_nodes,).fill()
    return random_forest.TensorForestEstimator(params, model_dir=model_dir)


model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
print('model directory = %s' % model_dir)
estimator = build_estimator(model_dir)
estimator = tensor_forest.RandomForestGraphs
#params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
#    num_classes=1, regression=True, num_features=392,
#    num_trees=FLAGS.num_trees, max_nodes=FLAGS.max_nodes).fill()
#graph = tf.Graph(estimator.graph_builder_class(params))
#sess = tf.Session(graph=graph)


def train_and_eval():
    """Train and evaluate the model."""
    X, Y = extract_data('train.csv')
    testX, testY = extract_data('test.csv')
    # TensorForest's LossMonitor allows training to terminate early if the
    # forest is no longer growing.
    early_stopping_rounds = 100
    check_every_n_steps = 100
    monitor = random_forest.LossMonitor(early_stopping_rounds,
                                        check_every_n_steps, )

    # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    estimator.fit(x=X, y=Y,
                  batch_size=FLAGS.batch_size, monitors=[monitor])

    results = estimator.evaluate(x=testX, y=testY,
                                 batch_size=1)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))
    for index, result in enumerate(estimator.predict(x=testX, as_iterable=True)):
        print(testY[index], ":", result)


def modelevaluation():
    testX, testY = extract_data('test.csv')
    results = estimator.evaluate(x=testX, y=testY,
                                 batch_size=FLAGS.batch_size)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))
    updownscore = 0
    score4bin = 0
    bad4bin = 0
    offbyone = 0
    predicted = []
    offbytwo = 0
    zerobound = .0001  # 0.000550902932254
    for index, result in enumerate(estimator.predict(x=testX, as_iterable=True)):
        result = 2.028445002 * result
        if testY[index] < 0:
            if testY[index] < -zerobound:
                real4bin = 0  # "bigdown"
            else:
                real4bin = 1  # "down"
        else:
            if testY[index] > zerobound:
                real4bin = 3  # "bigup"
            else:
                real4bin = 2  # "up"
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

        if pred4bin == real4bin:
            score4bin += 1
        if abs(pred4bin - real4bin) == 3:
            bad4bin += 1
        if abs(pred4bin - real4bin) == 2:
            offbytwo += 1
        if abs(pred4bin - real4bin) == 1:
            offbyone += 1

        if (testY[index] > 0) == (result > 0):
            updownscore += 1
        if index % 100 == 0: print(testY[index], ":", result, ":", real4bin, ":", pred4bin)
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


def runlive(tadata):
    tadata = np.matrix(tadata).astype(np.float32)
    # result = tf.sess.run(tadata)
    result = estimator.predict_proba(x=tadata, as_iterable=False, batch_size=None)
    result = 2.028445002 * result
    return result

def compiletree():
    testX, testY = extract_data('test.csv')
    compiled_predictor = compiledtrees.CompiledRegressionPredictor(estimator)
    predict = compiled_predictor.predict(testX)
    print(predict)


def main(_):
    train_and_eval()
    modelevaluation()
    #compiletree()


if __name__ == '__main__':
    tf.app.run()
