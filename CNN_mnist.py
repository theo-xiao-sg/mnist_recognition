# cnn model

from matplotlib import pyplot as plt
import joblib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from keras.datasets import mnist

import tensorflow as tf
tf.keras.utils.set_random_seed(1)

# to account for multiple times of MaxPooling2D of 2X2
image_size = 28

# load image dataset
def get_data_my_images():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# reshape dataset to meet the requirement of Conv2D
	x_train = x_train.reshape((x_train.shape[0], image_size, image_size, 1))
	x_test = x_test.reshape((x_test.shape[0], image_size, image_size, 1))

	# one-hot encode target column
	num_classes = 10
	y_train = to_categorical(y_train, num_classes)
	y_test = to_categorical(y_test, num_classes)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train = x_train/255
	x_test = x_test/255

	return x_train, x_test, y_train, y_test

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1), padding='same'))
	model.add(MaxPooling2D((2), padding='same'))

	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2), padding='same'))

	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2), padding='same'))

	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2), padding='same'))

	model.add(Flatten())

	model.add(Dense(10, activation='softmax'))

	# compile model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def plot_learning_progress(history):

	# plot accuracy
	plt.title('Accuracy')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='validation')
	plt.xlabel('Epoch')
	plt.xticks(range(0, len(history.history['loss']), 2))
	plt.xticks(rotation=45, ha='right')
	plt.ylabel('Accuracy')
	plt.tight_layout()
	plt.grid(True)
	plt.legend()

	# save plot to file
	plt.show(block=False)
	plt.savefig('learning_process_4layer_cnn.jpg')
	plt.close()

# load dataset
x_train, x_test, y_train, y_test = get_data_my_images()


# define model
model = define_model()

# define early stopping
early_stopping_monitor = EarlyStopping(patience=5, monitor='val_accuracy')

# fit model
history = model.fit(x_train, y_train, epochs=20, validation_split=0.2,
					callbacks=[early_stopping_monitor])

# evaluate model using the best model parameters saved
print('\n')
model.evaluate(x_test, y_test)

# learning curves
plot_learning_progress(history)

print('\n')
model.summary()

