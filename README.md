# mnist_recognition

 digits (mnist datasets) recognition using python based KNN and neural network algorithm
 
## Running Guide

This project is based on the Python programming language and primarily utilizes standard libraries like PIL, numpy, scikit-learn and os

### Environment Setup

Download the requirements.txt and install the required Python libraries. Please note all my 4 projects share the same requirements.txt. If you have done the installation for one project, you can skip it for the other 3 projects

```bash
# install all packages using requirements.txt
python -m pip install -r requirements.txt
```

### Training the Model

* If you want to train your model, you can run `KNN_model_training.py` or `neural_network_training.py` in the folder. The model will train itself using the 70000 over images in mnist. I have selected the best model parameters for you in the codes but you can always try yourselves with a set of parameters. For `KNN_model_training.py`, you can amend the line 22. For `neural_network_training.py`, you can amend the line 22.
* Then, the trained model file saved in a pickle file, either `handwriting_knn_minst.pkl` or `handwriting_neuralnetwork_minst.pkl`, will be generated in the folder.

### Try the recognition tool

Run the `recognize_numbers.py` to see the accuracy of the model.
