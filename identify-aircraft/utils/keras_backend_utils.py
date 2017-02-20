# Forcing backend to toggle between Theano and TensorFlow
def toggle_keras_backend():
    import importlib
    import json
    import keras
    import logging
    import os
    from pprint import pprint

    # The Keras configuration file is ~/.keras/keras.json,
    # according to https://keras.io/backend/
    _keras_dir = os.path.join(os.path.expanduser("~"), ".keras")
    _config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
    logging.info("Keras version : {}".format(keras.__version__))
    logging.info("Reading {} ...".format(_config_path))

    current_backend = "theano"
    if os.path.exists(_config_path) == True:
        with open(_config_path) as data_file:
            data = json.load(data_file)
            pprint(data)
            current_backend = data["backend"]

    toggled = { "theano":"tensorflow", "tensorflow":"theano" }
    toggled_backend = toggled[current_backend]
    image_dim_orderings = { "tensorflow":"tf", "theano":"th" }
    logging.info("Toggling from '{}' to '{}'".format(current_backend, toggled_backend))

    data = {"backend": toggled_backend,
            "epsilon": 1e-07,
            "floatx": "float32",
            "image_dim_ordering": image_dim_orderings[toggled_backend]
           }
    with open(_config_path, "w+") as fp:
        logging.info("Writing to {} ...".format(_config_path))
        json.dump(data, fp)

    importlib.reload(keras)
    from keras import backend
    importlib.reload(backend)
