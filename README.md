# Recognition of the MNIST dataset

Digits (mnist datasets) and fashion items (fashion_mnist datasets) recognition using python based KNN, neural network(NN), and convolutional neural network(CNN) algorithms
 
## Running Guide

This project is based on the Python programming language and primarily utilizes standard libraries like Tensorflow, PIL, numpy, scikit-learn and os

### Environment Setup

Download the requirements.txt and install the required Python libraries. Please note all my 4 projects share the same requirements.txt. If you have done the installation for one project, you can skip it for the other 3 projects

```bash
# install all packages using requirements.txt
python -m pip install -r requirements.txt
```

### Training the Model and check the accuracy on testing datasets

* If you want to train your model for MNIST datasets, you can run `KNN_mnist.py`, `neural_network_mnist.py`, or `CNN_mnist.py` in the folder. The model will train itself using the 60000 trainning images and then test itself using the 10000 testing images in MNIST datasets. I have selected a set of resonable model parameters for all the models in the codes but you can always try yourselves with a different set of parameters. For `KNN_mnist.py`, you can amend the line 25. For `neural_network_mnist.py`, you can amend the line 25.
* If you want to train your model for Fashion_MNIST datasets, you can run `KNN_fashion_mnist.py`, `neural_network_fashion_mnist.py`, or `CNN_fashion_mnist.py` in the folder. The model will train itself using the 60000 trainning images and then test itself using the 10000 testing images in Fashion_MNIST datasets. I have selected a set of resonable model parameters for all the models in the codes but you can always try yourselves with a different set of parameters.

