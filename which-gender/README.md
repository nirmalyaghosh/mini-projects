# Predicting Gender Based On Blog Text

A comparison of a few solutions for identifying the gender of blog authors based on his/her writing style.
It is based on a dataset containing 681288 blog posts downloaded from http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm.

- In [part 1](which_gender_part_1.ipynb), I use the *Bag of Words* approach with minimal preprocessing to compare a few classifiers.
- In [part 2](which_gender_part_2.ipynb), I do a bit more preprocessing and compare the *Bag of Words* approach with `Word2Vec` models on the blog text - both *continuous bag-of-word* (CBOW) and *skip-gram* (SG) models (hierarchical softmax and negative sampling).
- In [part 3](which_gender_part_3.ipynb), I compare 2 types of `Doc2Vec` models - both *distributed memory* (PV-DM) and *distributed bag of words* (PV-DBOW) models.

