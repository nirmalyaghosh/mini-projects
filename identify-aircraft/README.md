# Identify Aircraft Based On Photos

An attempt to identify aircraft models based on photos **without** relying on any additional information (i.e., no EXIF data, no tags, etc.). 

### Dataset
The photos were obtained from the Yahoo Flickr Creative Commons (YFCC) dataset, Wikipedia and other sources.

### Attempts

- In part 1, I train an image classifier (a convolutional neural network, aka CNN) to distinguish between photos of Airbus A321s and A340s. See [notebook](id_aircraft_script_1.ipynb) for network diagram. Validation accuracy 0.85-0.88, after 20 epochs. I use Keras with a Theano backend on a VirtualBox VM created using Vagrant.
