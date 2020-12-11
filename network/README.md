# Network module
---
> Implements the Convolutional Neural Network at the base of this project and the dataset class needed to handle the MNIST dataset in a proper way.

This module provides the classes needed for the implementation of the network.

The detailed explanation of the code used in this module is reported in the following Jupyter notebooks:

- [Dataset](https://github.com/filippoguerranti/handwritten_long_numbers_recognition/blob/dataset-handler-implementation/network/.dataset.ipynb)
- [CNN](...) **TODO**
- [Network](...) **TODO**


### Brief explanation

The `dataset` class has the role of handling the MNIST dataset. Its main features are:
- **download** the MNIST dataset from a url
- **read** the MNIST dataset and **load** it in a torch.tensor
- **save** the dataset in .pt format to be easily accessible within the PyTorch environment
- provide a method to create the dataset **splits**, according to some proportions
- provide a method to perform some **preprocessing** operations
