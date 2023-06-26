# Description: load mnist data and train a KNN model

import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
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

# tried different values of num_neighbors
# save num_neighbors and accuracy in a dictionary
result = {}
num_neighbors = [1, 3, 5, 7, 9]
# num_neighbors = [3]

for num_neighbor_i in num_neighbors:
    # create KNN model for each num_neighbors
    model = KNeighborsClassifier(n_neighbors=num_neighbor_i)
    print('model training for num_neighbors: {} ...'.format(num_neighbor_i))

    start = time.time()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    end = time.time()
    print("Time used for fitting and predicting using KNN with num_neighbors {} "
          "is {:.2f} seconds".format(num_neighbor_i, end - start))

    accuracy_i = accuracy_score(y_predict, y_test)
    result[num_neighbor_i] = accuracy_i
    print('num_neighbors: {}, accuracy: {:.2%}'.format(num_neighbor_i, accuracy_i))
    print('\n')

