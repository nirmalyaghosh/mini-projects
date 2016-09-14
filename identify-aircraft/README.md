# Identify Aircraft Based On Limited Number Of Photos

This is an attempt to train a deep neural network to identify different models of commercial aircraft based on very few photos - without relying on any additional information (i.e., no EXIF data, no tags, etc.). The underlying technique can be used for other image classification tasks where we have a small training dataset.

### Dataset
As of date, I have curated a dataset of 5920 photos of 39 different models of commercial aircraft 
([TSV](https://docs.google.com/spreadsheets/d/1zSUNhlpGDKtngK271UMJobtXcwojHiewczJ5FLpX4Es/pub?output=tsv), [Web](https://docs.google.com/spreadsheets/d/1zSUNhlpGDKtngK271UMJobtXcwojHiewczJ5FLpX4Es/pubhtml)). They were obtained from the Yahoo Flickr Creative Commons (YFCC) dataset, Wikipedia and other sources.

### Progress

- In [part 1](id_aircraft_script_1.ipynb), I train an image classifier (a convolutional neural network, aka CNN) to distinguish between photos of 4 different models of commercial aircraft. Validation accuracy **0.84**-**0.86**, after 20 epochs.
![](https://docs.google.com/drawings/d/1B7g5OCWWrKzFPE_hOj0yvVsN6PBxiov54MjmGW8FNYE/pub?w=959&h=429)
I use a  [VirtualBox VM with Keras and Theano installed](https://github.com/nirmalyaghosh/deep-learning-vm) - created using a Vagrant script (see [repository](https://github.com/nirmalyaghosh/deep-learning-vm)).
