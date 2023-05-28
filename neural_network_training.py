# Description: load mnist data and train a Neural Network Multi-layer Perceptron classifier model

import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time
import joblib


# load mnist data, toy dataset for image recognition
data = numpy.load('mnist.npz')
# 60000 training images
x_train = data['x_train']
y_train = data['y_train']
# 10000 testing images
x_test = data['x_test']
y_test = data['y_test']

# tried different values of number_of_neurons for one hidden layer
# save number_of_neurons and accuracy in a dictionary
result = {}
# number_of_neurons_list = [10, 50, 100, 200, 300, 400, 500, 600, 700]
number_of_neurons_list = [200]

for number_of_neurons_i in number_of_neurons_list:
    # create MLP model for one hidden layer
    model = MLPClassifier(hidden_layer_sizes=(number_of_neurons_i,))
    print('model training for hidden_layer_sizes: {} ...'.format(number_of_neurons_i))

    start = time.time()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    end = time.time()
    print("Time used for Neural Network MLP of one hidden layer with number_of_neurons {} is {:.2f} seconds".format(number_of_neurons_i, end - start))

    accuracy_i = accuracy_score(y_predict, y_test)
    result[number_of_neurons_i] = accuracy_i
    print('neural network model of one hidden layer with number_of_neurons: {}, '
          'accuracy: {:.2%}'.format(number_of_neurons_i, accuracy_i))
    print('\n')

# save model
joblib.dump(model,"neuralnetwork_minst.pkl")