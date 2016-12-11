# -*- coding: utf-8 -*-
"""
Trains the underlying model.

@author: Nirmalya Ghosh
"""

import logging
import os

import numpy as np

np.random.seed(
    371250)  # For reproducibility, needs to be set before Keras is loaded
from imp import reload
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

reload(logging)
logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")

train_data_dir = "data/train"
validation_data_dir = "data/test"

logging.info("Current PID : {}".format(os.getpid()))


def _get_data_generators(img_width, img_height, labels):
    train_datagen = ImageDataGenerator(
        fill_mode="nearest",
        horizontal_flip=True,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        batch_size=32,
        classes=labels,
        target_size=(img_width, img_height),
        class_mode="categorical")

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        batch_size=32,
        classes=labels,
        target_size=(img_width, img_height),
        class_mode="categorical")

    return train_generator, validation_generator


def _get_model(img_width, img_height, labels):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(labels)))
    model.add(Activation("sigmoid"))

    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    return model


def train_model_1(img_width, img_height, labels, name_prefix, nb_train_samples,
                  nb_validation_samples, nb_epoch=20):
    model = _get_model(img_width, img_height, labels)
    train_gen, valid_gen = _get_data_generators(img_width, img_height, labels)

    # Train
    logging.info("Training the model")
    model.fit_generator(
        train_gen,
        nb_epoch=nb_epoch,
        nb_val_samples=nb_validation_samples,
        samples_per_epoch=nb_train_samples,
        validation_data=valid_gen,
        verbose=2)
    logging.info("Done training the model")

    # Persist
    model_json = model.to_json()
    j_file_path = "{}_{}_epochs.json".format(name_prefix, nb_epoch)
    w_file_path = "{}_{}_epochs_weights.h5".format(name_prefix, nb_epoch)
    with open(j_file_path, "w") as jf:
        jf.write(model_json)
    model.save_weights(w_file_path)
    logging.info("Saved model and weights to {} and {}" \
                 .format(j_file_path, w_file_path))
