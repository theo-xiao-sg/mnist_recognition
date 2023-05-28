from PIL import Image
import numpy
import joblib
import os

# load model trained by mnist dataset
# model = joblib.load('knn_minst.pkl')
model = joblib.load('neuralnetwork_minst.pkl')

# minist image size is 28*28
image_size = 28

# load each image in images folder
for image in os.listdir('images'):

    imgs = Image.open('images/'+image)
    # read the figures from image file name and make them into a list
    num = image.split('.')[0]
    print('to recognise the number: {}'.format(num))

    # each image has 5 numbers
    number_predict = []
    for i in range(5):
        # crop each number to 100*100 image
        img = imgs.crop((i * 100, 0, (i + 1) * 100, 100))
        # resize image to 28*28 size since mnist dataset is 28*28
        img = img.resize((image_size, image_size))
        img_num = numpy.array(img)
        # convert image to 1D array since mnist dataset is 1D array
        img_num = img_num.reshape(1, image_size * image_size)
        # predict the number in the image
        pre = model.predict(img_num)
        number_predict.append(pre[0])

    # make a string of the predicted number
    number_predict_str = ''.join(str(i) for i in number_predict)
    print('the number predicted: {}'.format(number_predict_str))
    if number_predict_str == num:
        print('correct recognition of number ' + num)
    else:
        print('wrong recognition of number ' + num)
    print('\n')


