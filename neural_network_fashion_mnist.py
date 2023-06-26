# Description: load mnist data and train a Neural Network Multi-layer Perceptron classifier model

import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time
import joblib
from keras.datasets import fashion_mnist
from numpy.random import seed
seed(1)

image_size = 28

# load image dataset
def get_data_my_images():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # reshape the data to 1*784
    x_train = x_train.reshape(x_train.shape[0], image_size*image_size)
    x_test = x_test.reshape(x_test.shape[0], image_size*image_size)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train/255
    x_test = x_test/255

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = get_data_my_images()

# tried different values of number_of_neurons for one hidden layer
# save number_of_neurons and accuracy in a dictionary
result = {}
number_of_neurons_list = [100, 300, 500, 700, 900]
# number_of_neurons_list = [200]

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

