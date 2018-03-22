'''
    Michael Remington
    TF Learn Tutorial
    In this tutorial, we'll
    1. Load our data into pandas dataframes
    2. Convert categorical text data into one-hot vectors and numerical vectors
    3. Normalize continuous features
    4. Split our data into random train and dev sets
    5. Run a DNN
    6. Make a custom model with batch normalization
    7. Tune the model with random hyperparameter search
    8. Log our hyper parameter search to a sortable csv file
    Each section has an exit().
    Remove the exit() as you progress through each section.
    You'll turn in the csv of your hyperparameter search to canvas.
    For more tutorials and examples, see the TF Learn home page:
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/learn/python/learn
'''

# This tutorial will use TensorFlow, TF Learn, sklearn, and pandas
import os
import random

import pandas
import tensorflow as tf
import tensorflow.contrib.learn as learn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np

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

            label_mod = 0  # "bigdown"

        else:

            label_mod = 1  # "up"
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


    return fvecs_np, labels

'''
Part 1 : Load Data
------------------------------------------------------------------------------------------------------------------
'''

# First we'll load our raw data and labels into pandas dataframes
#data = pandas.read_csv('iris.data', delim_whitespace=True, header=None)
#labels = pandas.read_csv('iris.labels', delim_whitespace=True, header=None)

# Ignore this. For later one-hot example.
# text_labels = labels

data, labels = extract_data("normalizeddata.csv")
data = pandas.DataFrame(data, columns=list(range(88)))
labels = pandas.DataFrame(labels) #, columns=["bigdown", "down", "up", "bigup"])
labels_oneHot = labels
# Let's take a look at our data
print(data.shape)
print(labels.shape)

print(data.head())
print(labels.head())

# Our model needs to know the number of classes we're predicting
y_classes = 2

# Remove this exit to move on
# exit()

'''
Part 2 : Categorical Data Processing
------------------------------------------------------------------------------------------------------------------
'''

# Our labels are in text format and must be converted to integer values
#categorical_processor = learn.preprocessing.CategoricalProcessor()
#labels = pandas.DataFrame(categorical_processor.fit_transform(labels.values))

# Let's look at our new labels. Each category is now a number
#print(labels.head())

# Another way to handle categorical text or numbers is converting to categorical one-hot vectors
#labels_oneHot = pandas.get_dummies(text_labels)

# Let's look at our new one-hot labels. Each column now represents a category
#print(labels_oneHot.head())
# We won't use our one-hot labels for this experiment, but this is a common machine learning task

# Remove this exit to move on
# exit()

'''
Part 3 : Scaling Continuous Features
------------------------------------------------------------------------------------------------------------------
'''
# It's advantageous to scale continuous features to 0 mean and  unit standard deviation.
# Note: Don't scale categorical integers.

# Let's look at our unscaled data. You'll see that the mean and standard deviation are not 0 and 1 for any columns.
print(data.describe())
print(labels.describe())

# First we'll create a scaler
#scaler = preprocessing.StandardScaler()

# Now we'll scale (normalize) our data.
#data = scaler.fit_transform(data)
#data = pandas.DataFrame(data)

# Let's look at our scaled data. Now the mean and standard deviation are 0 and 1 for each column.
#print(data.describe())

# Now we'll split our data into randomly shuffled train, and dev sets.
# Setting a random seed allows for continuity between runs
X_train, X_dev, y_train, y_dev = train_test_split(data, labels, test_size=0.2, random_state=42)

# Remove this exit to move on
# exit()

'''
Part 4 : Simple DNN
------------------------------------------------------------------------------------------------------------------
'''
# Now we'll create a simple deep neural network tensorflow graph
# For regression you can use learn.TensorFlowDNNRegressor
classifier = learn.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=y_classes, batch_size=1, steps=100,
                                           optimizer="Adam", learning_rate=0.01, dropout=0.6)
                                           #,feature_columns=list(range(88)))

# learn.TensorFlowDNNClassifier

# Here we'll train our DNN
classifier.fit(X_train, y_train, logdir='/tmp/tf_learn/')

# and evaluate it on our dev data
predictions = classifier.predict(X_dev)
score = metrics.accuracy_score(y_dev, predictions)
print("Accuracy: %f" % score)

# Remove this exit to move on
# exit()

'''
Part 5 : TensorBoard
------------------------------------------------------------------------------------------------------------------
Running classifier.fit created the directory 'dnnLogs'.
Comment out the following line in part 3 above: data = scaler.fit_transform(data)
Run the DNN again. It will run without scaled features.
With your virtualenv activated, run the following command:
    tensorboard --logdir dnnLogs
Then go to localhost:6006 in your browser.
Click "Histograms", and then click 'X'.
You'll see two graphs of X. Notice that one is centered around zero and the other isn't. This is what scaling does.
Click "Graph" at the top to see a visual representation of the DNN TensorFlow graph we created.
Uncomment the line you commented out before so the features will be scaled again.
'''

'''
Part 6 : Custom Model
------------------------------------------------------------------------------------------------------------------
'''


# Next, we'll define a custom classfier model. You can use any TensorFlow or TF Learn code in this function.
def custom_model(X, y):
    # When running, X is a tensor of [batch size, num feats] and y is a tensor of [batch size, num outputs]

    # This model will use a technique called batch normalization
    # X = learn.ops.batch_normalize(X, scale_after_normalization=True)

    # Now we'll pass our normalized batch to a DNN
    # We can pass a TensorFlow object as the activation function
    layers = learn.ops.dnn(X, [10, 20, 10], activation=tf.nn.relu, dropout=0.5)

    # Given encoding of DNN, take encoding of last step (e.g hidden state of the
    # neural network at the last step) and pass it as features for logistic
    # regression over the label classes.
    return learn.models.logistic_regression(layers, y)


# We need a generic TF Learn model to wrap our custom model. For regression you can use learn.TensorFlowDNNRegressor.
classifier = learn.TensorFlowEstimator(model_fn=custom_model, n_classes=y_classes, batch_size=1, steps=500,
                                       optimizer="Adam", learning_rate=0.01)


# We'll make a function for training and evaluating
def run_model(classifier, logdir=None, monitor=None):
    # Train
    classifier.fit(X_train, y_train, logdir=logdir) #, monitors=monitor) #,  feature_columns=list(range(88))) #monitor=monitor,

    # Evaluate on dev data
    predictions = classifier.predict(X_dev)
    score = metrics.accuracy_score(y_dev, predictions)
    return score


score = run_model(classifier, '/tmp/customModelLogs')
print("Accuracy: %f" % score)

# We got 100% accuracy, but keep in mind our dataset has only 150 datapoints.

# Remove this exit to move on
# exit()

'''
Part 7 : Hyperparameter Search
------------------------------------------------------------------------------------------------------------------
'''


# Random sampling is a way to search for the optimal parameters for a model.
# It's recommended in 'Bengio. Practical Recommendations for Gradient-Based Training of Deep Architectures. 2012.'

# We'll make a function that will randomly generate hyper parameters or return set ones.
def getHyperparameters(tune=False):
    if tune:

        # Randomize DNN layers and hidden size
        hidden_units = []
        NUNITS = random.randrange(10, 100, step=10)
        NLAYERS = random.randint(1, 10)
        for layer in range(1, NLAYERS):
            hidden_units.append(NUNITS)

        # Make dict of randomized hyper params
        hyperparams = {
            'BATCH_SIZE': 1,
            'STEPS': 5000,
            'LEARNING_RATE': random.uniform(0.001, 0.09),
            'OPTIMIZER': random.choice(["SGD", "Adam", "Adagrad"]),
            'HIDDEN_UNITS': hidden_units,
            'NUM_LAYERS': NLAYERS,
            'NUM_UNITS': NUNITS,
            'ACTIVATION_FUNCTION': random.choice(
                [tf.nn.relu, tf.nn.tanh, tf.nn.relu6, tf.nn.elu, tf.nn.sigmoid, tf.nn.softplus]),
            'KEEP_PROB': random.uniform(0.5, 1.0),
            'MAX_BAD_COUNT': random.randrange(10, 1000, 10)
        }

    else:

        hidden_units = [10, 10, 10]

        hyperparams = {
            'BATCH_SIZE': 1,
            'STEPS': 1000,
            'LEARNING_RATE': 0.01,
            'OPTIMIZER': "Adam",
            'HIDDEN_UNITS': hidden_units,
            'NUM_LAYERS': len(hidden_units),
            'NUM_UNITS': hidden_units[0],
            'ACTIVATION_FUNCTION': tf.nn.relu,
            'KEEP_PROB': 0.6,
            'MAX_BAD_COUNT': random.randrange(10, 1000, 10)
        }

    return hyperparams


# Now we'll wrap our model in a function so that we can instantiate it with new hyper-parameters.
def instantiateModel(hyperparams):
    # We'll copy the same model from above
    def custom_model(X, y):
        #X = learn.ops.batch_normalize(X, scale_after_normalization=True) TODO possibly include this

        layers = learn.ops.dnn(X, hyperparams['HIDDEN_UNITS'], activation=hyperparams['ACTIVATION_FUNCTION'],
                               dropout=hyperparams['KEEP_PROB'])

        return learn.models.logistic_regression(layers, y)

    classifier = learn.TensorFlowEstimator(model_fn=custom_model, n_classes=y_classes,
                                           batch_size=hyperparams['BATCH_SIZE'],
                                           steps=hyperparams['STEPS'], optimizer=hyperparams['OPTIMIZER'],
                                           learning_rate=hyperparams['LEARNING_RATE'],)

    # We'll make a monitor so that we can implement early stopping based on our train accuracy. This will prevent overfitting.
    #monitor = learn.monitors.BaseMonitor(early_stopping_rounds=int(hyperparams['MAX_BAD_COUNT']),)#, print_steps=100)

    return classifier# , #monitor


# Now we'll 'tune' our model by running a hyper parameter search over many runs
for i in range(10000):

    hyperparams = getHyperparameters(tune=True)
    print(hyperparams)
    classifier = instantiateModel(hyperparams)

    score = run_model(classifier)  # , monitor=monitor)
    print("Accuracy: %f" % score)

    # We don't need to log this array
    del hyperparams['HIDDEN_UNITS']

    # Now we'll add the dev set accuracy to our dict
    hyperparams['dev_Accuracy'] = score

    # Convert the dict to a dataframe
    log = pandas.DataFrame(hyperparams, index=[0])
    print(log)

    # Write to a csv file
    csvName = 'model_log.csv'
    if not (os.path.exists(csvName)):
        # First run, write headers
        log.to_csv('model_log.csv', mode='a')
    else:
        log.to_csv('model_log.csv', mode='a', header=False)

# Open the csv file in an editor like libreoffice. Now you can sort thousands of runs by dev accuracy and find the best hyperparameters.

