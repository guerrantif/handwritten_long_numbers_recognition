# Segmentation module
---
> Implements the Graph-based image segmentation algorithm to segment the input image that will be then feed to the Convolutional Neural Network.

This module provides the classes needed for the implementation of the graph-based image segmentation algorithm proposed by Felzenszwalb et. al. ([paper](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf)).

The detailed explanation of the code used in this module is reported in the following Jupyter notebooks:

- [Graph-based image segmentation](...)
- [Digit extraction](...) **TODO**


### Brief explanation

The `ImageGraph` class ha the role of converting an input image into a graph and segment it using the graph-based segmentation algorithm. 

Its main features are:
- **preprocess** the input image
- **build** the graph from the input image
- **segment** the image according to the algorithm.

The whole process is carried out exploiting the `DisjoinSetForest` class, which handles the *disjoint-set forest* data structure.
