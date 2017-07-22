# -*- coding: utf-8 -*-
"""
A text classifier (using Keras) using pre-trained (GloVe) word embeddings.

@author: Nirmalya Ghosh
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, Flatten
from keras.models import load_model, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

logging.basicConfig(format="%(asctime)s %(processName)12s : %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _load_word_embeddings(config):
    file_path = config["word_embeddings_file_path"]
    logger.info("Loading word embeddings file, {}".format(file_path))
    embeddings_index = {}
    f = open(file_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def _predict(model, config):
    df = _read_dataset(config["test_dataset"], config)
    texts = df.text.values.tolist()
    data, w_index = _vectorize_texts(texts, config)
    logger.info("Making predictions based on {} tensor".format(data.shape))
    y = model.predict(data)
    # model.predict returns a list of lists,
    # where each element is a list of probabilities for each label

    # Mapping the index with highest probability to the correct label
    y = [np.argmax(elem) for elem in y]
    mapping = {}
    if os.path.exists(config["labels_mapping_file_path"]):
        with open(config["labels_mapping_file_path"]) as mapping_file:
            # mapping = json.load(mapping_file)
            mapping = {int(v): k for k, v in json.load(mapping_file).items()}
    y = [mapping[elem] for elem in y if elem in mapping]

    # Persist the predicted labels to indicated file
    df["pred"] = y
    df.to_csv(config["output_file_path"], encoding="utf-8-sig", index=False,
              sep="\t")

    # Compare to calculate percentage mismatch
    if "label" in df.columns.values.tolist():
        actuals = df.label.values.tolist()
        ctr = 0
        for i, val in enumerate(y):
            if val != actuals[i]:
                ctr += 1
        logger.info("{:.2f} % mismacth".format((ctr / len(y)) * 100))


def _prep_embedding_matrix(word_index, embeddings_index, config):
    num_words = min(config["max_num_words"], len(word_index))
    embedding_matrix = np.zeros((num_words, config["embedding_dimension"]))
    for word, i in word_index.items():
        if i >= num_words:
            # if i >= config["max_num_words"]:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, num_words


def _read_dataset(file_path, config):
    logger.info("Reading {}".format(file_path))
    cols = [config["label_column"], config["text_column"]]
    df = pd.read_csv(file_path, encoding="utf-8-sig",
                     error_bad_lines=False, index_col=False, sep="\t")
    col_names_renamed = {"" + config["label_column"]: "label",
                         "" + config["text_column"]: "text"}
    df.rename(columns=col_names_renamed, inplace=True)
    df["label"].fillna("", inplace=True)
    df["text"].fillna("", inplace=True)
    return df


def _train_model(config):
    logger.info("Preparing to train the classifier")
    df = _read_dataset(config["train_dataset"], config)
    # Vectorize the text into a 2D integer tensor
    texts = df.text.values.tolist()
    data, word_index = _vectorize_texts(texts, config)

    # Labels
    catenc = pd.factorize(df["label"])
    labels = catenc[0]
    labels = to_categorical(np.asarray(labels))
    df["temp"] = pd.factorize(df.label)[0]
    labels_index = dict(
        zip(df["label"].values.tolist(), df["temp"].values.tolist()))
    with open(config["labels_mapping_file_path"], "w") as fp:
        json.dump({k: str(v) for k, v in labels_index.items()}, fp)
    logger.info("Shape of both tensors: {}, {}"
                .format(data.shape, labels.shape))

    # Train-validation split
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(0.2 * data.shape[0])
    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    # Prepare the embedding matrix
    embeddings_index = _load_word_embeddings(config)
    embedding_matrix, num_words = \
        _prep_embedding_matrix(word_index, embeddings_index, config)

    # Load pre-trained (GloVe) word embeddings into an Embedding layer
    embedding_layer = Embedding(num_words,
                                config["embedding_dimension"],
                                weights=[embedding_matrix],
                                input_length=config["max_seq_length"],
                                trainable=False)

    # Train classifier
    logger.info("Training the classifier")
    sequence_input = Input(shape=(config["max_seq_length"],), dtype="int32")
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation="relu")(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation="relu")(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation="relu")(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    preds = Dense(len(labels_index), activation="softmax")(x)

    model = Model(sequence_input, preds)
    model.compile(loss="categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics=["acc"])

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=config["num_epochs"],
              validation_data=(x_val, y_val))
    logger.info("Saving model to {}".format(config["model_file_path"]))
    model.save(config["model_file_path"])
    return model


def _vectorize_texts(texts, config):
    tokenizer = Tokenizer(num_words=config["max_num_words"])
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=config["max_seq_length"])
    return data, word_index


def run_text_classifier(config_file_path):
    logger.info("Reading configurations from {}".format(config_file_path))
    with open(config_file_path, "r") as f:
        cfg = yaml.load(f)

    if os.path.exists(cfg["model_file_path"]) == False:
        model = _train_model(cfg)
    else:
        logger.info("Loading model from {}".format(cfg["model_file_path"]))
        model = load_model(cfg["model_file_path"])

    _predict(model, cfg)


if __name__ == "__main__":

    if len(sys.argv[1:]) < 1:
        logger.error("Configuration file not specified")
        sys.exit(0)

    run_text_classifier(sys.argv[1])
