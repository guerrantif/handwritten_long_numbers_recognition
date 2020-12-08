# Recognition of handwritten (long) numbers
> Neural Network recognizer of long handwritten numbers via the use of a webcam.

The aim of this project is to build a Neural Network capable to recognize long numbers (composed of several digits) written by hand. 

---
* [How it works](#how-it-works)
* [Download and setup](#download-and-setup)
* [Usage example](#usage-example)
* [History](#history)
* [Directory structure](#directory-structure)
* [References](#references)
* [Info](#info)

---

## How it works

**Workflow**
![](img/workflow.png)

> **TODO**: here goes the explanation of the entire project 

## Download and Setup

You can download the project directory using the following commands in your Linux/MacOS/Windows terminal:

* `git clone https://github.com/filippoguerranti/handwritten_long_digit_recognition.git`
* `cd handwritten_long_number_recognition`

After downloading the folder, you can type:

* `pip3 install -r requirements.txt`

This command will install all the needed dependencies for this project.
Some issues may arise for the OpenCv library. If it happens, please see the note below for more informations.

> **NOTE**: informations about how to install OpenCV in your platform can be found [here][opencv-installation].

## Usage example

Some usage example can be found in the `notebook` folder, in the form of Jupyter notebooks.

* `0_introduction.ipynb`: 

## History

* _2020/12/07_
  * `cnn` class built 
* _2020/12/03_
  * project starts
  * first tests using openCV
   
   
## Directory structure

```
├── img
│   ├── logo_unisi.jpg
│   ├── numbers.jpg
│   ├── stuff.jpg
│   └── workflow.png
├── LICENSE
├── notebook1.ipynb
├── README.md
├── references
│   ├── 1412.6980.pdf
│   ├── 1502.01852.pdf
│   ├── 1506.02025.pdf
│   ├── 1710.05381.pdf
│   └── 2001.09136.pdf
├── requirements.txt
├── src
│   └── prova.py
├── tests
│   ├── install_opencv.py
│   ├── MNIST_dataset.py
│   ├── step-by-step.py
│   └── __utils__.py
└── __utils__
    └── cnn.py
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
