<!--![](img/logo_unisi.jpg) -->

# Recognition of handwritten (long) numbers
> Neural Network recognizer of long handwritten numbers via the use of a webcam.

The aim of this project is to build a Neural Network capable to recognize long numbers (composed of several digits) written by hand. 

---
* [How it works](#how-it-works)
* [Download](#download)
* [Development Setup](#development-setup)
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

## Download

You can download the project directory using the following commands in your Linux/MacOS/Windows terminal:

```sh
git clone https://github.com/filippoguerranti/handwritten_long_digit_recognition.git
cd handwritten_long_number_recognition
```

## Development setup

After downloading the directory, make sure all the dependencies are installed.

To check if everything is correctly set run the following command:

```sh
python3 requirements_check.py
```

You should get something like:

```sh
All the dependecies are correctly installed.
```

If something is missig, you will be notified by an error message.

> **NOTE**: informations about how to install OpenCV in your platform can be found [here][opencv-installation].

## Usage example

In order to start the application, do the following

```sh
python3 main.py
```


## History

* _2020/12/03_
   * project starts
   * first tests using openCV
   
   
## Directory structure

```
handwritten_long_numbers_recognition
├── img
│   ├── logo_unisi.jpg
│   ├── numbers.jpg
│   └── stuff.jpg
├── LICENSE
├── README.md
└── src
    └── prova.py
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
