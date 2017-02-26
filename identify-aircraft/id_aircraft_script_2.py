# -*- coding: utf-8 -*-
"""
Applies transfer learning using the pre-trained Inception v3 model to classify
images of commercial aircraft.

This script is based on https://www.tensorflow.org/tutorials/image_retraining
and the example at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py.

A pre-trained Inception v3 model (from https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip)
is loaded, then the topmost layer is removed and a new one is trained on the
images of commercial aircraft.

None of the classes of images of commercial aircraft in my dataset are amongst
the 1000 classes the Inception v3 network was trained on.

To the best of my knowledge none of the images in my dataset appear amongst the
images in the ImageNet dataset.

Hyper-parameters and other configurations are set via identify_aircraft.yml.

This script saves the model if validation accuracy is better than previous
(or incomplete) runs and restores from it (if available).

Some notes:
- A 'Bottleneck' is an informal term used to refer to the layer just before
the final output layer that actually does the classification. This layer is a
compact summary of the images, since it has to contain enough information for
the classifier to make a good choice in a very small set of values.
[Reference : https://www.tensorflow.org/tutorials/image_retraining#bottlenecks]

@author: Nirmalya Ghosh (and authors of the original script)
"""

import glob
import hashlib
import os.path
import random
import re
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

tf.logging.set_verbosity(tf.logging.INFO)

BOTTLENECK_TENSOR_NAME = "pool_3/_reshape:0"
BOTTLENECK_TENSOR_SIZE = 2048
JPEG_DATA_TENSOR_NAME = "DecodeJpeg/contents:0"
KEY = "script_2"
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
RESIZED_INPUT_TENSOR_NAME = "ResizeBilinear:0"


def __add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.
    Args:
      result_tensor: The new final node that produces results.
      ground_truth_tensor: The node we feed ground truth data into.
    Returns:
      Tuple of (evaluation step, prediction).
    """
    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope("accuracy"):
            evaluation_step = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", evaluation_step)
    return evaluation_step, prediction


def __add_final_training_ops(class_count, bottleneck_tensor, config):
    """Adds a new softmax and fully-connected layer for training.
    We need to retrain the top layer to identify our new classes, so this
    function adds the right operations to the graph, along with some variables
    to hold the weights, and then sets up all the gradients for the backward
    pass. The set up for the softmax and fully-connected layers is based on:
    https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
    Args:
      class_count: Integer of how many categories of things to recognize.
      bottleneck_tensor: The output of the main CNN graph.
      config:
    Returns:
      The tensors for the training and cross entropy results, and tensors for
      the bottleneck input and ground truth input.
    """
    with tf.name_scope("input"):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
            name="BottleneckInputPlaceholder")

        ground_truth_input = tf.placeholder(tf.float32,
                                            [None, class_count],
                                            name="GroundTruthInput")

    # Organizing the following ops as `final_training_ops` so they're easier
    # to see in TensorBoard
    layer_name = "final_training_ops"
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            layer_weights = tf.Variable(
                tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count],
                                    stddev=0.001), name="final_weights")
            __variable_summaries(layer_weights)
        with tf.name_scope("biases"):
            layer_biases = tf.Variable(tf.zeros([class_count]),
                                       name="final_biases")
            __variable_summaries(layer_biases)
        with tf.name_scope("Wx_plus_b"):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram("pre_activations", logits)

    final_tensor_name = config[KEY]["final_tensor_name"]
    # final_tensor_name is name of the new final node that produces results
    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram("activations", final_tensor)

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits)
        with tf.name_scope("total"):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("cross_entropy", cross_entropy_mean)

    learning_rate = config[KEY]["learning_rate"]
    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(learning_rate) \
            .minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input,
            ground_truth_input, final_tensor)


def __add_input_distortions(config):
    """Creates the operations to apply the specified distortions.
    During training it can help to improve the results if we run the images
    through simple distortions like crops, scales, and flips. These reflect the
    kind of variations we expect in the real world, and so can help train the
    model to cope with natural data more effectively. Here we take the supplied
    parameters and construct a network of operations to apply them to an image.
    Cropping
    ~~~~~~~~
    Cropping is done by placing a bounding box at a random position in the full
    image. The cropping parameter controls the size of that box relative to the
    input image. If it's zero, then the box is the same size as the input and 
    no cropping is performed. If the value is 50%, then the crop box will be 
    half the width and height of the input. In a diagram it looks like this:
    <       width         >
    +---------------------+
    |                     |
    |   width - crop%     |
    |    <      >         |
    |    +------+         |
    |    |      |         |
    |    |      |         |
    |    |      |         |
    |    +------+         |
    |                     |
    |                     |
    +---------------------+
    Scaling
    ~~~~~~~
    Scaling is a lot like cropping, except that the bounding box is always
    centered and its size varies randomly within the given range. For example
    if the scale percentage is zero, then the bounding box is the same size as
    the input and no scaling is applied. If it's 50%, then the bounding box 
    will be in a random range between half the width and height and full size.
    Args:
      config: top level configuration object
    Returns:
      The jpeg input layer and the distorted result tensor.
    """

    flip_left_right = config[KEY]["distort_images"]["flip_left_right"]
    random_crop = config[KEY]["distort_images"]["random_crop"]
    random_scale = config[KEY]["distort_images"]["random_scale"]
    random_brightness = config[KEY]["distort_images"]["random_brightness"]

    jpeg_data = tf.placeholder(tf.string, name="DistortJPGInput")
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                           minval=1.0,
                                           maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
    precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d,
                                   [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH,
                                    MODEL_INPUT_DEPTH])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=brightness_min,
                                         maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name="DistortResult")
    return jpeg_data, distort_result


def __cache_bottlenecks(sess, image_lists, images_dir, bottlenecks_dir,
                        jpeg_data_tensor, bottleneck_tensor):
    """Ensures all the training, testing, and validation bottlenecks are cached.
    Because we're likely to read the same image multiple times (if there are no
    distortions applied during training) it can speed things up a lot if we
    calculate the bottleneck layer values once for each image during
    preprocessing, and then just read those cached values repeatedly during
    training. Here we go through all the images we've found, calculate those
    values, and save them off.
    Args:
      sess: The current active TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      images_dir: Root folder string of the subfolders containing the training
      images.
      bottlenecks_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: Input tensor for jpeg data from file.
      bottleneck_tensor: The penultimate output layer of the graph.
    Returns:
      Nothing.
    """
    ctr = 0  # keeps track of the number of bottlenecks created
    __ensure_dir_exists(bottlenecks_dir)
    for label_name, label_lists in image_lists.items():
        for category in ["training", "testing", "validation"]:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                __get_or_create_bottleneck(sess, image_lists, label_name,
                                           index,
                                           images_dir, category,
                                           bottlenecks_dir,
                                           jpeg_data_tensor, bottleneck_tensor)

                ctr += 1
                if ctr % 100 == 0:
                    tf.logging.info("%s bottleneck files created.", ctr)


def __create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                             images_dir, category, sess, jpeg_data_tensor,
                             bottleneck_tensor):
    image_path = \
        __get_image_path(image_lists, label_name, index, images_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal("File does not exist %s", image_path)
    image_data = gfile.FastGFile(image_path, "rb").read()
    bottleneck_values = __run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    bottleneck_string = ",".join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, "w") as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def __create_image_lists(config):
    """Builds a list of training images from the file system.
    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.
    Args:
      config: top level configuration object
    Returns:
      A dictionary containing an entry for each label sub-folder, with images
      split into training, testing, and validation sets within each label.
    """

    images_dir = config[KEY]["images_dir"]
    percentages = config[KEY]["percentages"]
    testing_percentage = percentages["testing"]
    validation_percentage = percentages["validation"]
    train_on_labels = config[KEY]["train_on_labels"]

    if not gfile.Exists(images_dir):
        tf.logging.info("Image directory '%s' not found.", images_dir)
        return None

    result = {}
    extensions = ["jpg", "jpeg", "JPG", "JPEG"]
    sub_dirs = [x[0] for x in gfile.Walk(images_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == images_dir:
            continue
        if dir_name not in train_on_labels:
            continue

        tf.logging.info("Looking for images in '%s'", dir_name)
        for extension in extensions:
            file_glob = os.path.join(images_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            tf.logging.info("No files found")
            continue
        if len(file_list) < 20:
            tf.logging.warn("%s has < 20 images, may cause issues.", dir_name)
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print(
                "WARNING: Folder {} has more than {} images. Some images will "
                "never be selected.".format(dir_name,
                                            MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a
            # way of grouping photos that are close variations of each other.
            # For example this is used in the plant disease data set to group
            # multiple pictures of the same leaf.
            hash_name = re.sub(r'_nohash_.*$', "", file_name)
            # This looks a bit magical, but we need to decide whether this file
            # should go into the training, testing, or validation sets, and we
            # want to keep existing files in the same set even if more files
            # are subsequently added.
            # To do that, we need a stable way of deciding based on just the
            # file name itself, so we do a hash of that and then use that to
            # generate a probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(
                compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (
                        testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            "dir": dir_name,
            "training": training_images,
            "testing": testing_images,
            "validation": validation_images,
        }
    return result


def __create_inception_graph(config):
    """"Creates a graph from saved GraphDef file and returns a Graph object.
    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
    """
    with tf.Session() as sess:
        model_filepath = config[KEY]["pretrained_model"]
        with gfile.FastGFile(model_filepath, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name="", return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def __ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def __get_formatted_timestamp_str(config):
    return datetime.fromtimestamp(time.time()).strftime(
        config[KEY]["timestamp_format"])


def __get_bottleneck_path(image_lists, label_name, index, bottlenecks_dir,
                          category):
    """"Returns a path to a bottleneck file for a label at the given index.
    Args:
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      bottlenecks_dir: Folder string holding cached files of bottleneck values.
      category: Name string of set to pull images from - training, testing, or
      validation.
    Returns:
      File system path string to an image that meets the requested parameters.
    """
    return __get_image_path(image_lists, label_name, index, bottlenecks_dir,
                            category) + ".txt"


def __get_image_path(image_lists, label_name, index, images_dir, category):
    """"Returns a path to an image for a label at the given index.
    Args:
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Int offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      images_dir: Root folder string of subfolders containing training images.
      category: Name string of set to pull images from - training, testing, or
      validation.
    Returns:
      File system path string to an image that meets the requested parameters.
    """
    if label_name not in image_lists:
        tf.logging.fatal("Label does not exist %s.", label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal("Category does not exist %s.", category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal("Label %s has no images in the category %s.",
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists["dir"]
    full_path = os.path.join(images_dir, sub_dir, base_name)
    return full_path


def __get_model_name(config, step_id, validation_accuracy):
    return "{}_{}_step_{}_val_accuracy_{:.0f}" \
        .format(config[KEY]["model_name"],
                __get_formatted_timestamp_str(config), step_id,
                validation_accuracy)


def __get_or_create_bottleneck(sess, image_lists, label_name, index,
                               images_dir, category, bottlenecks_dir,
                               jpeg_data_tensor, bottleneck_tensor):
    """Retrieves or calculates bottleneck values for an image.
    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.
    Args:
      sess: The current active TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be modulo-ed by the
      available number of images for the label, so it can be arbitrarily large.
      images_dir: Root folder string  of the subfolders containing the training
      images.
      category: Name string of which  set to pull images from - training,
      testing, or validation.
      bottlenecks_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: The tensor to feed loaded jpeg data into.
      bottleneck_tensor: The output tensor for the bottleneck values.
    Returns:
      Numpy array of values produced by the bottleneck layer for the image.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists["dir"]
    sub_dir_path = os.path.join(bottlenecks_dir, sub_dir)
    __ensure_dir_exists(sub_dir_path)
    bottleneck_path = __get_bottleneck_path(image_lists, label_name, index,
                                            bottlenecks_dir, category)
    if not os.path.exists(bottleneck_path):
        __create_bottleneck_file(bottleneck_path, image_lists, label_name,
                                 index,
                                 images_dir, category, sess, jpeg_data_tensor,
                                 bottleneck_tensor)
    with open(bottleneck_path, "r") as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(",")]
    except:
        tf.logging.error("Invalid float found, recreating bottleneck")
        did_hit_error = True

    if did_hit_error:
        __create_bottleneck_file(bottleneck_path, image_lists, label_name,
                                 index,
                                 images_dir, category, sess, jpeg_data_tensor,
                                 bottleneck_tensor)
        with open(bottleneck_path, "r") as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here,
        # since they shouldn't happen after a fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(",")]

    return bottleneck_values


def __get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                    bottlenecks_dir, images_dir,
                                    jpeg_data_tensor,
                                    bottleneck_tensor):
    """Retrieves bottleneck values for cached images.
    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.
    Args:
      sess: Current TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      how_many: If positive, a random sample of this size will be chosen.
      If negative, all bottlenecks will be retrieved.
      category: Name string of which set to pull from - training, testing, or
      validation.
      bottlenecks_dir: Folder string holding cached files of bottleneck values.
      images_dir: Root folder string of the subfolders containing the training
      images.
      jpeg_data_tensor: The layer to feed jpeg image data into.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.
    Returns:
      List of bottleneck arrays, their corresponding ground truths, and the
      relevant filenames.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = __get_image_path(image_lists, label_name, image_index,
                                          images_dir, category)
            bottleneck = __get_or_create_bottleneck(sess, image_lists,
                                                    label_name,
                                                    image_index, images_dir,
                                                    category,
                                                    bottlenecks_dir,
                                                    jpeg_data_tensor,
                                                    bottleneck_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = __get_image_path(image_lists, label_name,
                                              image_index,
                                              images_dir, category)
                bottleneck = __get_or_create_bottleneck(sess, image_lists,
                                                        label_name,
                                                        image_index,
                                                        images_dir,
                                                        category,
                                                        bottlenecks_dir,
                                                        jpeg_data_tensor,
                                                        bottleneck_tensor)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


def __get_random_distorted_bottlenecks(
        sess, image_lists, how_many, category, images_dir, input_jpeg_tensor,
        distorted_image, resized_input_tensor, bottleneck_tensor):
    """Retrieves bottleneck values for training images, after distortions.
    If we're training with distortions like crops, scales, or flips, we have to
    recalculate the full model for every image, and so we can't use cached
    bottleneck values. Instead we find random images for the requested category,
    run them through the distortion graph, and then the full graph to get the
    bottleneck results for each.
    Args:
      sess: Current TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      how_many: The integer number of bottleneck values to return.
      category: Name string of which set of images to fetch - training, testing,
      or validation.
      images_dir: Root folder string of the subfolders containing the training
      images.
      input_jpeg_tensor: The input layer we feed the image data to.
      distorted_image: The output node of the distortion graph.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.
    Returns:
      List of bottleneck arrays and their corresponding ground truths.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = __get_image_path(image_lists, label_name, image_index,
                                      images_dir,
                                      category)
        if not gfile.Exists(image_path):
            tf.logging.fatal("File does not exist %s", image_path)
        jpeg_data = gfile.FastGFile(image_path, "rb").read()
        # Note that we materialize the distorted_image_data as a numpy array
        # before sending running inference on the image. This involves 2
        # memory copies and might be optimized in other implementations.
        distorted_image_data = sess.run(distorted_image,
                                        {input_jpeg_tensor: jpeg_data})
        bottleneck = __run_bottleneck_on_image(sess, distorted_image_data,
                                               resized_input_tensor,
                                               bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


def main(config):
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(config[KEY]["summaries_dir"]):
        tf.gfile.DeleteRecursively(config[KEY]["summaries_dir"])
    tf.gfile.MakeDirs(config[KEY]["summaries_dir"])

    # Set up the pre-trained graph
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = \
        (__create_inception_graph(config))

    # Look at the folder structure, and create lists of all the images
    images_dir = config[KEY]["images_dir"]
    percentages = config[KEY]["percentages"]
    image_lists = __create_image_lists(config)
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error("No valid folders of images found at %s", images_dir)
        return -1
    if class_count == 1:
        tf.logging.error("Require multiple image folders at %s. Only 1 found",
                         images_dir)
        return -1

    do_distort_images = __should_distort_images(config)

    sess = tf.Session()

    if do_distort_images:
        # We will be applying distortions, so setup the operations we'll need
        distorted_jpeg_data_tensor, distorted_image_tensor = \
            __add_input_distortions(config)
    else:
        # We'll make sure we've calculated the 'bottleneck' image summaries and
        # cached them on disk
        bottlenecks_dir = config[KEY]["bottlenecks_dir"]
        __cache_bottlenecks(sess, image_lists, images_dir,
                            bottlenecks_dir, jpeg_data_tensor,
                            bottleneck_tensor)

    # Add the new layer that we'll be training
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = __add_final_training_ops(class_count,
                                              bottleneck_tensor,
                                              config)

    # Create the operations we need to evaluate the accuracy of our new layer
    evaluation_step, prediction = \
        __add_evaluation_step(final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()
    summaries_dir = config[KEY]["summaries_dir"]
    train_writer = tf.summary.FileWriter(summaries_dir + "/train", sess.graph)
    validation_writer = tf.summary.FileWriter(summaries_dir + "/validation")

    # Set up all our weights to their initial default values
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # Run the training for as many cycles as indicated in the YML config file
    bottlenecks_dir = config[KEY]["bottlenecks_dir"]
    how_many_training_steps = config[KEY]["num_training_steps_before_ending"]
    train_batch_size = config[KEY]["train_batch_size"]
    best_validation_accuracy = 0
    best_model_file_pattern = None
    models_dir = config[KEY]["models_dir"]

    # Check latest checkpoint (best model from the previous/incomplete run)
    # and restore from it (if available)
    ckpt = tf.train.get_checkpoint_state(models_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = ckpt.model_checkpoint_path
        tf.logging.info("Latest checkpoint : %s", ckpt_path)
        saver.restore(sess, ckpt_path)
        tf.logging.info("Restored from : %s", ckpt_path)
        best_model_file_pattern = ckpt_path.split(os.path.sep)[-1]
        best_validation_accuracy = int(best_model_file_pattern.split("_")[-1])
        tf.logging.info("Previous best validation accuracy : %.1f%%",
                        best_validation_accuracy)

    for i in range(how_many_training_steps):
        # Get a batch of input bottleneck values, either calculated fresh every
        # time with distortions applied, or from the cache stored on disk
        if do_distort_images:
            train_bottlenecks, train_ground_truth = \
                __get_random_distorted_bottlenecks(sess,
                                                   image_lists,
                                                   train_batch_size,
                                                   "training",
                                                   images_dir,
                                                   distorted_jpeg_data_tensor,
                                                   distorted_image_tensor,
                                                   resized_image_tensor,
                                                   bottleneck_tensor)
        else:
            train_bottlenecks, train_ground_truth, _ = \
                __get_random_cached_bottlenecks(sess,
                                                image_lists,
                                                train_batch_size,
                                                "training",
                                                bottlenecks_dir,
                                                images_dir,
                                                jpeg_data_tensor,
                                                bottleneck_tensor)

        # Feed the bottlenecks and ground truth into the graph, and run a
        # training step. Capture training summaries for TensorBoard with the
        # `merged` op
        fetches = [merged, train_step]
        feed_dict = {bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth}
        train_summary, _ = sess.run(fetches, feed_dict=feed_dict)
        train_writer.add_summary(train_summary, i)

        # Every so often, print out how well the graph is training.
        is_last_step = (i + 1 == how_many_training_steps)
        eval_step_interval = config[KEY]["eval_step_interval"]
        if (i % eval_step_interval) == 0 or is_last_step:
            fetches = [evaluation_step, cross_entropy]
            feed_dict = {bottleneck_input: train_bottlenecks,
                         ground_truth_input: train_ground_truth}
            train_accuracy, cross_entropy_value = \
                sess.run(fetches, feed_dict=feed_dict)
            print("%s: Step %d: Train accuracy = %.1f%%" % (
                datetime.now(), i, train_accuracy * 100))
            print('%s: Step %d: Cross entropy = %f' % (
                datetime.now(), i, cross_entropy_value))

        validation_batch_size = config[KEY]["validation_batch_size"]
        validation_bottlenecks, validation_ground_truth, _ = \
            (__get_random_cached_bottlenecks(sess,
                                             image_lists,
                                             validation_batch_size,
                                             "validation",
                                             bottlenecks_dir,
                                             images_dir,
                                             jpeg_data_tensor,
                                             bottleneck_tensor))

        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op
        fetches = [merged, evaluation_step]
        feed_dict = {bottleneck_input: validation_bottlenecks,
                     ground_truth_input: validation_ground_truth}
        validation_summary, validation_accuracy = \
            sess.run(fetches, feed_dict=feed_dict)
        validation_accuracy *= 100
        validation_writer.add_summary(validation_summary, i)
        print("%s: Step %d: Validation accuracy = %.1f%% (N=%d)" %
              (datetime.now(), i, validation_accuracy,
               len(validation_bottlenecks)))

        # Save the model (get rid of the older file generated in this run)
        if validation_accuracy > best_validation_accuracy:
            model_name = __get_model_name(config, i, validation_accuracy)
            model_path = os.path.join(models_dir, model_name)
            saver.save(sess, model_path)
            best_validation_accuracy = validation_accuracy
            if best_model_file_pattern:
                tf.logging.info("Deleting file(s) matching %s",
                                best_model_file_pattern)
                for f in glob.glob(os.path.join(models_dir, "{}.*"
                        .format(best_model_file_pattern))):
                    os.remove(f)
            # Keep track of the last saved model
            best_model_file_pattern = model_name

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    test_batch_size = config[KEY]["test_batch_size"]
    test_bottlenecks, test_ground_truth, test_filenames = \
        (__get_random_cached_bottlenecks(sess,
                                         image_lists,
                                         test_batch_size,
                                         "testing",
                                         bottlenecks_dir,
                                         images_dir,
                                         jpeg_data_tensor,
                                         bottleneck_tensor))

    fetches = [evaluation_step, prediction]
    feed_dict = {bottleneck_input: test_bottlenecks,
                 ground_truth_input: test_ground_truth}
    test_accuracy, predictions = sess.run(fetches, feed_dict=feed_dict)
    test_accuracy *= 100
    print("Final test accuracy = %.1f%% (N=%d)" %
          (test_accuracy, len(test_bottlenecks)))

    if config[KEY]["print_misclassified_test_images"]:
        try:
            print("=== MISCLASSIFIED TEST IMAGES ===")
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                    print("%70s  %s" % (
                        test_filename, image_lists.keys()[predictions[i]]))
        except Exception as exc:
            print(str(exc))

    # Write out the trained graph and labels with weights stored as constants
    output_labels = config[KEY]["output_labels"]
    output_graph_filepath = os.path.join(config[KEY]["models_dir"],
                                         "identify_aircraft_graph_def.pb")

    output_graph_def = \
        graph_util.convert_variables_to_constants(sess,
                                                  graph.as_graph_def(),
                                                  [final_tensor_name])
    with gfile.FastGFile(output_graph_filepath, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    with gfile.FastGFile(output_labels, "w") as f:
        f.write("\n".join(image_lists.keys()) + "\n")


def __run_bottleneck_on_image(sess, image_data, image_data_tensor,
                              bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer
    Args:
      sess: Current active TensorFlow Session.
      image_data: String of raw JPEG data.
      image_data_tensor: Input data layer in the graph.
      bottleneck_tensor: Layer before the final softmax.
    Returns:
      Numpy array of bottleneck values.
    """
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def __should_distort_images(config):
    """Whether any distortions are enabled, from the input flags.
    Args:
      config: top level configuration object
    Returns:
      Boolean value indicating whether any distortions should be applied.
    """
    flip_left_right = config[KEY]["distort_images"]["flip_left_right"]
    random_crop = config[KEY]["distort_images"]["random_crop"]
    random_scale = config[KEY]["distort_images"]["random_scale"]
    random_brightness = config[KEY]["distort_images"]["random_brightness"]

    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
            (random_brightness != 0))


def __variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)


if __name__ == "__main__":
    with open("identify_aircraft.yml", "r") as f:
        config = yaml.load(f)
    main(config)
