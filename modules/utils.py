""" modules.utils.py
Summary
-------
This module contains some utility functions used by other modules and by the main function.

Functions
---------
webcam_capture()
    opens the webcam of the user and allows to capture a frame

save_image_steps()
    saves (as images) all the segmentation steps that an input image passes through

main_args_parser()
    argument parser of the main function
    used by hlnr.py

classify()
    starts the classification execution by calling modules.cnn.CNN.classify() with the appropriated arguments

train()
    starts the training execution by calling modules.cnn.CNN.train_cnn() with the appropriated arguments

eval()
    starts the evaluation execution by calling modules.cnn.CNN.eval_cnn() with the appropriated arguments


Copyright 2021 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# custom modules
from .cnn import *
from .dataset import *
from .segmentation import *

import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from datetime import datetime



def webcam_capture() -> (np.ndarray, str):
    """ Opens the webcam and allows the user to take an image. 

    Returns
    -------
    image: np.ndarray
        image captured by the webcam

    image_path: str
        relative path to where the captured images has been saved

    Notes
    -----
    Strongly based on `openCV`.
    """

    # opens a camera for video capturing and assigns it a name
    cam = cv2.VideoCapture(index=0)
    cv2.namedWindow(winname="SPACE: snapshot | ESC: exit")

    while True:

        retval, frame = cam.read()
        if not retval:
            print("Unable to grab the frame!")
            break

        cv2.imshow(winname="SPACE: snapshot | ESC: exit", mat=frame)
        k = cv2.waitKey(delay=1)

        # ESC is pressed
        if k % 256 == 27:
            print("Exiting...")
            image = None
            break
        
        # SPACE is pressed 
        elif k % 256 == 32:
            image = frame
            break

    cam.release()
    cv2.destroyAllWindows()

    # Save the image 
    if image is not None:
        now = datetime.now()
        image_name = "img-{}.png".format(now.strftime("%Y%m%d-%H%M%S"))
        image_dir = "img/webcam/"
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        image_path = os.path.join(image_dir, image_name)

        cv2.imwrite(image_path, image) 

    return image, image_path



def save_image_steps(
    image_path: str, 
    segmented: GraphBasedSegmentation
    ) -> None:
    """ Saves the images of all the segmentation steps taken by the input image.  

    When an image is segmented by the GraphBasedSegmentation object (see modules.segmentation.GraphBasedSegmentation class)
    it takes several steps:
    * it is segmented (segmented_img)
    * boxes are drawn around digits (boxed_img)
    * digits are extracted (digits)

    Parameters
    ----------
    image_path: str
        path of the input image
    
    segmented: GraphBasedSegmentation
        GraphBasedSegmentation object of the input image

    Notes
    -----
    """

    #region - segmented image
    segmented_image = segmented.segmented_img
    segmented_path = '{}-segmented.png'.format(image_path[:-4])
    segmented_image.save(segmented_path)
    print("\n\nSegmented image saved: {}".format(segmented_path))
    #endregion

    #region - boxed image
    boxed_image = segmented.boxed_img
    boxed_path = '{}-boxed.png'.format(image_path[:-4])
    boxed_image.save(boxed_path)
    print("Boxed image saved: {}".format(boxed_path))
    #endregion

    #region - extracted digits image
    digits_path = '{}-digits.png'.format(image_path[:-4])
    fig = plt.figure(figsize=(3*len(segmented.digits),3))
    for i in range(len(segmented.digits)):
        image = segmented.digits[i][0]
        sp = fig.add_subplot(1, len(segmented.digits), i+1)
        plt.axis('off')
        plt.imshow(image, cmap='gray')
    plt.savefig(digits_path)
    print("Digits image saved: {}\n".format(digits_path))
    #endregion



def main_args_parser() -> argparse.Namespace:
    """ Implements an easy user-friendly command-line interface.  

    It creates three main subparser (classify, train, eval) and add the appropriated arguments for each subparser.

    Returns
    -------
    args: argparse.Namespace
        input arguments provided by the user    
    """
    #region - MAIN parser
    parser = argparse.ArgumentParser(description='Handwritten long number recognition')

    subparsers = parser.add_subparsers(dest='mode', 
                                        help='<required> program execution mode: classify with a pre-trained model or re-train the model', 
                                        required=True)
    #endregion

    #region - CLASSIFY subparser 
    parser_classify = subparsers.add_parser('classify', 
                                            help='classify an input image using the pre-trained model', 
                                            description='CLASSIFY mode: classify an input image using a pre-trained model')

    image_from = parser_classify.add_mutually_exclusive_group()

    image_from.add_argument('-f', '--folder', 
                            type=str, 
                            help='input image from folder, if not specified from webcam', 
                            metavar='PATH_TO_IMAGE', 
                            default=None)

    which_model = parser_classify.add_mutually_exclusive_group()

    which_model.add_argument('-a', '--augmentation', 
                            action='store_true', 
                            help='use model trained WITH data augmentation')

    which_model.add_argument('-m', '--model', 
                            type=str, 
                            help='user custom model from path', 
                            metavar='PATH_TO_MODEL')

    parser_classify.add_argument('-d', '--device', 
                                type=str, 
                                help='(default=cpu) device to be used for computations {cpu, cuda:0, cuda:1, ...}', 
                                default='cpu')
    #endregion

    #region - TRAIN subparser
    parser_train = subparsers.add_parser('train', 
                                        help='re-train the model in your machine and save it to reuse in classify phase', 
                                        description='TRAIN mode: re-train the model in your machine and save it to reuse in classify phase')
    
    parser_train.add_argument('-a', '--augmentation', 
                            action='store_true', 
                            help='set data-augmentation procedure ON (RandomRotation and RandomResizedCrop)')
    
    parser_train.add_argument('-s', '--splits', 
                            nargs=2, 
                            type=float, 
                            help='(default=[0.7,0.3]) proportions for the dataset split into training and validation set', 
                            default=[0.7,0.3], 
                            metavar=('TRAIN', 'VAL'))
    
    parser_train.add_argument('-b', '--batch_size', 
                            type=int, 
                            help='(default=64) mini-batch size', 
                            default=64)
    
    parser_train.add_argument('-e', '--epochs', 
                            type=int, 
                            help='(default=10) number of training epochs', 
                            default=10)
    
    parser_train.add_argument('-l', '--learning_rate', 
                            type=float, 
                            help='(default=10) learning rate', 
                            default=0.001)
    
    parser_train.add_argument('-w', '--num_workers', 
                            type=int, 
                            help='(default=3) number of workers', 
                            default=3)

    parser_train.add_argument('-d', '--device', 
                            type=str, 
                            help='(default=cpu) device to be used for computations {cpu, cuda:0, cuda:1, ...}', 
                            default='cpu')
    #endregion
    
    #region - EVAL subparser
    parser_eval = subparsers.add_parser('eval', 
                                        help='evaluate the model accuracy on the test set of MNIST', 
                                        description='EVAL mode: evaluate the model accuracy on the test set of MNIST')

    parser_eval.add_argument('model', 
                            type=str, 
                            help='<required> path to the model to be evaluated', 
                            metavar='PATH_TO_MODEL')

    parser_eval.add_argument('-d', '--device', 
                            type=str, 
                            help='(default=cpu) device to be used for computations {cpu, cuda:0, cuda:1, ...}', 
                            default='cpu')
    #endregion
    
    args = parser.parse_args()

    return args



def classify(
    image_path: str or None,
    model: str or None,  
    augmentation: bool, 
    device: str
    ) -> None:
    """ Takes an image as input, segments it in single digits and classifies each digit.

    If image_path is None, it takes an image from the webcam.  
    It loads the defined model as a classifier (default or user-specified).  
    The image is segmented using the graph-based segmentation algorithm.  
    The model classifies each single digit extracted from the segmented image.

    Parameters
    ----------
    image_path: str or None
        path to input image (if str)
        image capturing from webcam (if None)

    model: str or None
        path to the model to use as classifier (if str)
        uses the default pre-trained model (if None)
    
    augmentation: bool
        uses the default model trained with data augmentation (if True)
        uses the default model trained without data augmentation (if False)

    device: str
        represents the device {cpu, cuda:0, ...} in which the computation is performed

    Notes
    -----
    Strongly based on `modules.utils.webcam_capture()`, `modules.cnn` and `modules.segmentation`.
    """
    
    #region - image loading or capturing
    if image_path is not None:
        if os.path.exists(image_path) and os.path.isfile(image_path):
            image = image_path
        else:
            raise ValueError("Wrong image path.")
    else:
        image, image_path = webcam_capture()
        if image is None:
            raise RuntimeError("Unable to take webcam image. When window appears press SPACE to take a snapshot.")
    #endregion

    #region - model loading
    classifier = CNN(device)
    
    if model is not None:
        classifier.load(model)
    else:
        if augmentation:
            classifier.load('models/CNN-128b-60e-0.0001l-a.pth')
        else:
            classifier.load('models/CNN-128b-60e-0.0001l.pth')
    #endregion

    #region - image segmentation
    segmented = GraphBasedSegmentation(image)
    segmented.segment(k=4500, min_size=100, preprocessing=True)

    segmented.generate_image()
    segmented.digits_boxes_and_areas()
    segmented.extract_digits()

    save_image_steps(image_path, segmented)
    #endregion

    #region - classification
    output = classifier.classify(segmented.digits)
    output = ''.join(str(digit.item()) for digit in output)
    print('\n\nThe recognized number is: {}\n\n'.format(output))
    #endregion



def train(
    augmentation: bool, 
    splits: list, 
    batch_size: int, 
    epochs: int, 
    lr: float, 
    num_workers: int, 
    device: str
    ) -> None:
    """ Prepares the MNIST dataset and trains the CNN on the dataset based on the input parameters.

    The MNIST dataset is downloaded from the official source (if not present in the `data/` directory).  
    The training, validation and test sets are created (according with `splits`).  
    The classifier is prepared and it is then trained (calling `modules.cnn.CNN.train_cnn()` and according with the other parameters).  
    After the training phase, the model is evaluated (calling `modules.cnn.CNN.eval_cnn()`).  
    
    Parameters
    ----------
    augmentation: bool
        applies data augmentation techniques to the dataset samples during training phase (if True)
        uses the default dataset samples without any preprocessing during training phase (if False)

    splits: list
        training and validation proportions for the split procedure (ex. [0.7, 0.3])

    batch_size: int
        number of samples to be used as mini-batches during training (ex. 64)

    epochs: int
        number of epochs for the training phase (ex. 10)

    lr: float
        learning rate used during the update of the network weights (ADAM optimizer is used)

    num_workers: int
        number of processes to be used while loading the dataset in training phase

    device: str
        represents the device {cpu, cuda:0, ...} in which the computation is performed

    Notes
    -----
    Strongly based on `modules.dataset` and `modules.cnn`.
    """

    #region - classifier preparation
    classifier = CNN(data_augmentation=augmentation, device=device)
    #endregion

    #region - dataset preparation
    print("\n\nDataset preparation ...\n")

    basedir = os.path.dirname(sys._getframe(1).f_globals['__file__'])
    dataset_folder = os.path.join(basedir, 'data/')
    print("Dataset folder: {}".format(dataset_folder))

    training_validation_set = MNIST(folder=dataset_folder, train=True, download_dataset=True, empty=False)

    test_set = MNIST(folder=dataset_folder, train=False, download_dataset=True, empty=False)

    training_set, validation_set = training_validation_set.splits(proportions=splits, shuffle=True)
  
    # setting preprocessing operations on training samples (if augmentation==True)
    training_set.set_preprocess(classifier.preprocess)

    print("\n\nStatistics: training set\n")
    training_set.statistics()
    print("\n\nStatistics: validation set\n")
    validation_set.statistics()
    print("\n\nStatistics: test set\n")
    test_set.statistics()
    #endregion

    # directory in which models will be saved
    model_path = os.path.join(basedir, 'models/')


    #region - classifier training
    print("\n\nTraining phase...\n")
    classifier.train_cnn(training_set=training_set, validation_set=validation_set, batch_size=batch_size, 
                        lr=lr, epochs=epochs, num_workers=num_workers, model_path=model_path)
    #endregion
    
    #region - classifier evaluation
    print("\n\nValidation phase...\n")

    model_name = '{}CNN-{}b-{}e-{}l{}.pth'.format(model_path, batch_size, epochs, lr, '-a' if augmentation else '')
    classifier.load(model_name)

    training_acc = classifier.eval_cnn(training_set)
    validation_acc = classifier.eval_cnn(validation_set)
    test_acc = classifier.eval_cnn(test_set)

    print("\n\nAccuracies\n")
    print("training set:\t{:.2f}".format(training_acc))
    print("validation set:\t{:.2f}".format(validation_acc))
    print("test set:\t{:.2f}".format(test_acc))
    print("\n\nModel path: {}\n".format(model_name))



def eval(
    model_path: str, 
    device: str
    ) -> None:
    """ Evaluates the given model over the MNIST test set.

    Parameters
    ----------
    model_path: str
        path to the model to be evaluated
    
    device: str
        represents the device {cpu, cuda:0, ...} in which the computation is performed
    """
    #region - dataset preparation
    print("\n\nDataset preparation ...\n")
    basedir = os.path.dirname(sys._getframe(1).f_globals['__file__'])
    dataset_folder = os.path.join(basedir, 'data/')
    print("Dataset folder: {}".format(dataset_folder))
    
    test_set = MNIST(folder=dataset_folder, train=False, download_dataset=True, empty=False)
    #endregion
                

    #region - model loading and evaluation
    print("\n\nEvaluation phase...\n")

    classifier = CNN(device=device)
    classifier.load(model_name)

    test_acc = classifier.eval_cnn(test_set)

    print("\ntest set accuracy:\t{:.2f}\n\n".format(test_acc))
    #endregion