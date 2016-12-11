# -*- coding: utf-8 -*-
"""
Predicts the primary subject's name, based on the trained model.

@author: Nirmalya Ghosh
"""

import logging

import numpy as np

np.random.seed(371250)  # Needs to be set before Keras is loaded
from imp import reload
from keras.models import model_from_json
from keras.preprocessing import image as image_utils

reload(logging)
logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")


def load_model(model_file_path, model_weights_file_path):
    logging.info("Loading model from {} ...".format(model_file_path))
    json_file = open(model_file_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights_file_path)
    logging.info("Loaded {} from disk".format(model_file_path))
    return loaded_model


def predict_person_in_photo(photo_file_path, model, names):
    img = image_utils.load_img(photo_file_path, target_size=(125, 125))
    img_data = [np.array(img)]
    img_data = np.array(img_data, dtype=np.uint8)
    img_data = img_data.transpose((0, 3, 1, 2))
    img_data = img_data.astype("float32")
    img_data = img_data / 255  # img shape is (1, 3, 125, 125)
    # Predict the name
    preds = model.predict(img_data, batch_size=1, verbose=2).tolist()[0]
    names_with_preds = sorted(list(zip(names, preds)), key=lambda x: -x[1])
    return names_with_preds
