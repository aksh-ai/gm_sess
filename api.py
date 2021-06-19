import os
import io
import cv2
import math
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from flask import Flask, jsonify, request, send_file
from tensorflow.keras.layers import Input, Dense, Reshape, BatchNormalization, Activation, Conv2DTranspose, Dropout, Conv2D, MaxPool2D, Flatten

app = Flask(__name__)

LATENT_SIZE = 100
IMAGE_SIZE = 28

def build_classifier():
    i = Input((IMAGE_SIZE, IMAGE_SIZE, 1))

    x = Conv2D(32, 3, padding='same', activation='relu')(i)
    x = Conv2D(32, 3, activation='relu')(x)
    x = MaxPool2D(2, 2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, 3, padding='same', activation='relu')(i)
    x = Conv2D(64, 3, activation='relu')(x)
    x = MaxPool2D(2, 2)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=i, outputs=x)

    return model

def build_generator():
    input_shape = (LATENT_SIZE, )

    image_resize = IMAGE_SIZE // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    inputs = Input(shape=input_shape, name='z_input')

    x = Dense(image_resize * image_resize * layer_filters[0])(inputs)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters, kernel_size, strides, padding='same')(x)

    x = Activation('sigmoid')(x)
    
    generator = Model(inputs, x, name='generator')
    
    return generator

def plot_images():
    z = np.random.uniform(-1.0, 1.0, size=[1, LATENT_SIZE])
    images = generator.predict(z)
    pred = model.predict(images)
    pred = np.argmax(pred, axis=1)[0]
    image = np.reshape(images[0], [IMAGE_SIZE, IMAGE_SIZE])
    image = cv2.resize(image, (300, 300))
    plt.imsave("static/img/1.png", image, cmap='gray')
    return str(pred)

generator = build_generator()
generator.load_weights("models/dcgan_mnist.h5")
print(generator.summary())

model = build_classifier()
model.load_weights("models/mnist_classifier.h5")
print(model.summary())

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    return jsonify({'class_name': plot_images()})

if __name__ == '__main__':
  app.run(debug=True, port=5000)