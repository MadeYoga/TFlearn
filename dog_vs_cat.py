import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = 'G:/Programs/Python/DogVsCat/train'
TEST_DIR = 'G:/Programs/Python/DogVsCat/test'
IMG_SIZE = 50
learning_rate = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}'.format(learning_rate, '6conv-basic')

## EXTRACT DATA
## [0, 1] 0catness 1dogness, [1, 0]

def label_img(img):
    ## dog.93.png, -3 -> dog
    word_label = img.split('.')[-3]
    if word_label == 'cat' :
        return [1, 0]
    elif word_label == 'dog' :
        return [0, 1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(
            cv2.imread(path, cv2.IMREAD_GRAYSCALE),
            (IMG_SIZE, IMG_SIZE)
            )
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(
            cv2.imread(path, cv2.IMREAD_GRAYSCALE),
            (IMG_SIZE, IMG_SIZE)
            )
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data

## train_data = create_train_data()
## or if already have train data:
train_data = np.load('train_data.npy')

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

## 2 -> dog or cat
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(
    convnet,
    optimizer='adam',
    learning_rate=learning_rate,
    loss='categorical_crossentropy',
    name='targets'
    )

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)): ## LOAD SAVED MODEL / CHECK POINT
     model.load(MODEL_NAME)
     print('model loaded!')
     
train = train_data[:-500]
test = train_data[-500:]

X = np.array([data[0] for data in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # data[0] pixel data
y = [data[1] for data in train]

X_test = np.array([data[0] for data in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [data[1] for data in test] ## label

model.fit(
    {'input': X},
    {'targets': y},
    n_epoch=5,
    validation_set=( {'input': X_test}, {'targets': y_test} ), 
    snapshot_step=500,
    show_metric=True,
    run_id=MODEL_NAME
    )

model.save(MODEL_NAME)

import matplotlib.pyplot as plt

## test_data = process_test_data()
## if already have test data :
test_data = np.load('test_data.npy')

fig = plt.figure()

for num, data in enumerate (test_data[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num+1)## subplot starts at 1
    og = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label='Dog'
    else:
        str_label='Cat'

    y.imshow(og, cmap='gray')
    plt.title(str_label)

    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

## tensorboard --logdir=foo:G:\Programs\Python\DogVsCat\log (on cmd)
    


