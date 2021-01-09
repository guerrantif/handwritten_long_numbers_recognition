# Recognition of handwritten (long) numbers
> Neural Network recognizer of long handwritten numbers via the use of a webcam.

The aim of this project is to build a CNN model trained on MNIST dataset and to exploit its classification capabilities to recognize a sequence of several single handwritten digits (that can be considered as a long number) given as an input image that the user can take from her/his webcam.

In this project there is an extensive use of the following (but not only) Python libraries:
* PyTorch
* Pillow
* OpenCV

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

[workflow]: img/workflow.png "Project workflow"

As the picture shows, the project may be divided into two main sub-problems:

 * CNN building and training phase
 * Webcam image segmentation
 
Having the trained model and the correct segmentation of the input image, the digits classification task and the handwritten long number recognition one, are trivial problems.

### CNN building and training phase

This subproblem is developed in the `network` module and has the following structure:
1. **Download the MNIST dataset and decode it** (_a fully detailed description of this phase is provided in [this](https://github.com/filippoguerranti/handwritten_long_numbers_recognition/blob/main/network/file_decoding_procedure.ipynb) notebook_)
   * `network.utils.download()` function: downloads the `.IDX` file from the given URL source and stores it in the folder given as argument.
   * `network.utils.store_file_to_tensor()` function: takes the downloaded file (format `.IDX`) and store its contents into a `torch.tensor` following the provided encoding.
2. **Build a class to handle the dataset**
   * `network.dataset.MNIST()` class: takes care of:
     * downloading the dataset
     * storing it into `data` and `labels` tensors
     * splitting the dataset into training set and validation set according to some proportions
     * returning a `DataLoader` of the current dataset (needed for iterating over it)
     * printing some statistics and classes distribution
     * applying some preprocessing operations (such as random rotations for data augmentation)
3. **Build the CNN model and train it on the MNIST dataset**
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
4. **Obtain the trained model**



> **TODO**: here goes the explanation of the entire project 

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
* [NumPy documentation][numpy]
* [PyTorch documentation][torch]
* [OpenCV documentation][opencv]


## Info

Author: Filippo Guerranti <filippo.guerranti@student.unisi.it>

I am a M.Sc. student in Computer and Automation Engineering at [University of Siena][unisi], [Department of Information Engineering and Mathematical Sciences][diism]. This project is inherent the Neural Network course held by prof. [Stefano Melacci][melacci].

For any suggestion or doubts please contact me by email.

Distributed under the Apache-2.0 License. _See ``LICENSE`` for more information._

Link to this project: [https://github.com/filippoguerranti/handwritten_long_numbers_recognition][project]


<!-- Markdown link & img dfn's -->
[wiki]: https://github.com/filippoguerranti/handwritten_long_digit_recognition/wiki
[mnist]: http://yann.lecun.com/exdb/mnist/
[numpy]: https://numpy.org/doc/stable/
[torch]: https://pytorch.org/docs/stable/index.html
[opencv]: https://docs.opencv.org/master/index.html
[opencv-installation]: https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html
[project]: https://github.com/filippoguerranti/handwritten_long_numbers_recognition
[unisi]: https://www.unisi.it/
[diism]: https://www.diism.unisi.it/it
[melacci]: https://www3.diism.unisi.it/~melacci/index.html
