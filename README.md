# Recognition of handwritten (long) numbers
> Neural Network recognizer of long handwritten numbers via the use of a webcam.

The aim of this project is to build a model based on CNN and image segmentation capable to recognize long numbers (composed of several digits) written by hand. 

The main tools used for this projects are:
* PyTorch
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
![](img/workflow.png)

As shown in the previous picture the project may be divided into two main sub-problems:

 * CNN building and training phase
 * Webcam image segmentation
 
Having the trained model and the correct segmentation of the input image, the digits classification task and the handwritten long number recognition one, are trivial problems.

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

* _2020/12/26_
  * `ImageGraph` class built
  * `DisjointSetForest` class built
* _2020/12/12
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
