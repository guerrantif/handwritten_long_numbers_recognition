# Recognition of handwritten (long) numbers
> Neural Network recognizer of long handwritten numbers via the use of a webcam.

The aim of this project is to build a CNN model trained on MNIST dataset and to exploit its classification capabilities to recognize a sequence of several single handwritten digits (that can be considered as a long number) given as an input image that the user can take from her/his webcam.

---
* [Project description](#project-description)
* [Download and setup](#download-and-setup)
* [Usage example](#usage-example)
* [History](#history)
* [Directory structure](#directory-structure)
* [References](#references)
* [Info](#info)
---

## Project description

**Workflow**

![workflow][workflow]

As the picture shows, the project may be divided into three main sub-problems:

 * [CNN building and training phase](#cnn-building-and-training-phase)
 * [Webcam image segmentation](#webcam-image-segmentation)
 * [Number recognition](#number-recognition)
 
Having the trained model and the correct segmentation of the input image, the digits classification task and the handwritten long number recognition one, are trivial problems.



### CNN building and training phase

This phase is developed in the `network` module and has the following structure:
1. **MNIST dataset download and decoding** (_a fully detailed description of this phase is provided in [this][file-decode-notebook] notebook_)
   * `network.utils.download()` function: downloads the `.IDX` file from the given URL source and stores it in the folder given as argument.
   * `network.utils.store_file_to_tensor()` function: takes the downloaded file (format `.IDX`) and store its contents into a `torch.tensor` following the provided encoding.
2. **Dataset class building**
   * `network.dataset.MNIST()` class: takes care of:
     * downloading the dataset
     * storing it into `data` and `labels` tensors
     * splitting the dataset into training set and validation set according to some proportions
     * returning a `DataLoader` of the current dataset (needed for iterating over it)
     * printing some statistics and classes distribution
     * applying some preprocessing operations (such as random rotations for data augmentation)
3. **CNN model building and training on the MNIST dataset**
   * `network.cnn.CNN()` class: takes care of:
     * building the CNN model (shown in the picture below)
     * defining the preprocess operations to be performed on the elements of the dataset while iterating over it
     * saving and loading the model
     * implementing the:
       * `forward()` function: implicitly build the computational-graph
       * `__decision()` function: chooses the output neuron having maximum value among all the others 
       * `__loss()` function: applies the Cross Entropy loss to the output (before softmax)
       * `__performance()` function: computes the accuracy as number of correct decisions divided by the total number of samples
     * training the model by mean of the `train_cnn()` method (Adam optimizer is the default one)
     * evaluating the model by mean of the `eval_cnn()` method

![cnn-model][cnn-model]

### Webcam image segmentation

The image segmentation and the webcam capture tasks are implemented in the `input` module.
The structure of this module is as follows:
* **Webcam image capture**
  * TODO
* **Graph-based image segmentation** (a detailed explanation of the algorithm is given [here][graph-based-segmentation])
  * `input.segmentation.GraphBasedSegmentation()` class: implements the graph-based segmentation algorithm proposed by Felzenszwalb et. al. ([paper][graph-based-segmentation-paper]).
    * builds a graph from an input image
    * segments the image applying the Felzenszwalb's algorithm to the graph
    * finds the segmented regions' boundaries
    * draws the boxes around the segmented regions
  * `input.segmentation.DisjointSetForest()` class: the data-structure used by the algorithm (not really used outside the other class).
* **Digits extraction** (a detailed explanation is given [here][digits-extraction])


### Number recognition


## Download and Setup

First, be sure to have installed in your system at least the following:

* `python 3.8.5`
* `pip 20.0.2`
* `git 2.25.1`

I'm not able to guarantee that other versions will work correctly.

Then, the project directory can be downloaded using the following commands in a Linux/MacOS/Windows terminal:

* `git clone https://github.com/filippoguerranti/handwritten_long_numbers_recognition.git`
* `cd handwritten_long_number_recognition`

After downloading the folder, you can type:

* `pip3 install -r requirements.txt`

This command will install all the needed dependencies for this project.
Some issues may arise for the OpenCv library. If it happens, please see the note below for more informations.

> **NOTE**: informations about how to install OpenCV in your platform can be found [here][opencv-installation].

## Usage example

The `usage_example.ipynb` notebook shows some simple usage cases.

## History

* _2020/12/28_
  * Webcam image capture
  * Drawn boxes around digits and digit extraction
* _2020/12/26_
  * `GraphBasedSegmentation` class built
* _2020/12/25_
  * `DisjointSetForest` class built
* _2020/12/12_
  * `dataset` class handler built (custom class)
* _2020/12/08_
  * Training and testing procedure completed (model1: 99.0% accuracy on test set)
  * `MnistDataset` class built from (`torchvision.Datasets`) - **Deprecated**
* _2020/12/07_
  * `CNN` class built 
* _2020/12/03_
  * first tests using `openCV` - **Deprecated**
  * project starts
   
   
## Directory structure

```
.
├── img
│   ├── numbers.jpg
│   ├── results
│   │   └── CNN-model1-10_epochs-2000_batchsize.png
│   └── workflow.png
├── __init__.py
├── input
│   ├── __init__.py
│   └── segmentation.py
├── LICENSE
├── models
│   └── CNN-model1.pth
├── network
│   ├── cnn.py
│   ├── dataset.py
│   └── __init__.py
├── notebook
├── README.md
├── references
│   ├── 1412.6980.pdf
│   ├── 1502.01852.pdf
│   ├── 1506.02025.pdf
│   ├── 1710.05381.pdf
│   └── 2001.09136.pdf
├── requirements.txt
└── usage_example.ipynb
```
  
## References

* [MNIST dataset][mnist]
* [PyTorch documentation][torch]
* [Pillow documentation][pillow]
* [OpenCV documentation][opencv]
* [NumPy documentation][numpy]


## Info

Author: Filippo Guerranti <filippo.guerranti@student.unisi.it>

I am a M.Sc. student in Computer and Automation Engineering at [University of Siena][unisi], [Department of Information Engineering and Mathematical Sciences][diism]. This project is inherent the Neural Network course held by prof. [Stefano Melacci][melacci].

For any suggestion or doubts please contact me by email.

Distributed under the Apache-2.0 License. _See ``LICENSE`` for more information._

Link to this project: [https://github.com/filippoguerranti/handwritten_long_numbers_recognition][project]


<!-- Markdown link & img dfn's -->
[workflow]: img/workflow.png
[file-decode-notebook]: https://github.com/filippoguerranti/handwritten_long_numbers_recognition/blob/main/network/file_decoding_procedure.ipynb
[cnn-model]: img/cnn-model.png
[graph-based-segmentation]: https://github.com/filippoguerranti/handwritten_long_numbers_recognition/blob/main/input/graph_based_segmentation.ipynb
[graph-based-segmentation-paper]: http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf
[digits-extraction]: https://github.com/filippoguerranti/handwritten_long_numbers_recognition/blob/main/input/digits_extraction.ipynb
[mnist]: http://yann.lecun.com/exdb/mnist/
[numpy]: https://numpy.org/doc/stable/
[pillow]: https://pillow.readthedocs.io/en/stable/
[torch]: https://pytorch.org/docs/stable/index.html
[opencv]: https://docs.opencv.org/master/index.html
[opencv-installation]: https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html
[project]: https://github.com/filippoguerranti/handwritten_long_numbers_recognition
[unisi]: https://www.unisi.it/
[diism]: https://www.diism.unisi.it/it
[melacci]: https://www3.diism.unisi.it/~melacci/
