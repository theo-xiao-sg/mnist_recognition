# Description: load mnist data and train a KNN model

import numpy
from sklearn.neighbors import KNeighborsClassifier
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

# tried different values of num_neighbors
# save num_neighbors and accuracy in a dictionary
result = {}
# num_neighbors = [1, 3, 5, 7, 9]
num_neighbors = [3]

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

# save model
joblib.dump(model,"knn_minst.pkl")