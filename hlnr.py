"""
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

import modules.utils as utils
import modules.cnn as cnn
import modules.dataset as dataset
import modules.segmentation as seg
import cv2
import os
import matplotlib.pyplot as plt
import argparse



def save_digits(digits, path):

    fig = plt.figure(figsize=(30,15))
    for i in range(len(digits)):
        image = digits[i][0]
        sp = fig.add_subplot(3, len(digits), i+1)
        plt.axis('off')
        plt.imshow(image, cmap='gray')
    plt.savefig(path)



def main_args_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(description='Handwritten long number recognition')

    subparsers = parser.add_subparsers(dest='mode'
                                    , help='<required> program execution mode: classify with the pre-trained model or re-train the model'
                                    , required=True)

    # create the parser for the "CLASSIFY" command 
    parser_classify = subparsers.add_parser('classify'
                                            , help='classify an input image using the pre-trained model'
                                            , description='CLASSIFY mode: classify an input image using the pre-trained model')

    image_from = parser_classify.add_mutually_exclusive_group()
    image_from.add_argument('-w', '--webcam'
                            , action='store_true'
                            , help='input image from webcam')

    image_from.add_argument('-f', '--folder'
                            , type=str
                            , help='input image from folder'
                            , metavar='PATH_TO_IMAGE')

    parser_classify.add_argument('-a', '--augmentation'
                                , action='store_true'
                                , help='use model trained WITH data augmentation')

    parser_classify.add_argument('-d', '--device'
                                , type=str
                                , help='(default=cpu) device to be used for computations {cpu, cuda:0, cuda:1, ...}'
                                , default='cpu')

    # create the parser for the "TRAIN" command
    parser_train = subparsers.add_parser('train'
                                        , help='re-train the model in your machine and save it to reuse in classify phase'
                                        , description='TRAIN mode: re-train the model in your machine and save it to reuse in classify phase')
    
    parser_train.add_argument('-a', '--augmentation'
                            , action='store_true'
                            , help='set data-augmentation procedure ON (RandomRotation and RandomResizedCrop)')
    
    parser_train.add_argument('-s', '--splits'
                            , nargs=2
                            , type=float
                            , help='(default=[0.8,0.2]) proportions for the dataset split into training and validation set'
                            , default=[0.8,0.2]
                            , metavar=('TRAIN', 'VAL'))
    
    parser_train.add_argument('-b', '--batch_size'
                            , type=int
                            , help='(default=64) mini-batch size'
                            , default=64)
    
    parser_train.add_argument('-e', '--epochs'
                            , type=int
                            , help='(default=10) number of training epochs'
                            , default=10)
    
    parser_train.add_argument('-l', '--learning_rate'
                            , type=float
                            , help='(default=10) learning rate'
                            , default=0.001)
    
    parser_train.add_argument('-w', '--num_workers'
                            , type=int
                            , help='(default=3) number of workers'
                            , default=3)

    parser_train.add_argument('-d', '--device'
                            , type=str
                            , help='(default=cpu) device to be used for computations {cpu, cuda:0, cuda:1, ...}'
                            , default='cpu')
    
    return parser.parse_args()



def classify(webcam, image_path, augmentation, device):

    # image capture
    # ------------------------
    if webcam:
        image = utils.webcam_capture()
    # ------------------------
    
    if image_path is not None:
        image = image_path

    # creating a new classifier
    # ------------------------
    classifier = cnn.CNN(device)
    # ------------------------

    # load pre-trained model
    # ------------------------
    if augmentation:
        classifier.load('models/CNN-batch_size150-lr0.001-epochs40-a.pth')
    else:
        classifier.load('models/CNN-batch_size150-lr0.001-epochs40.pth')
    # ------------------------

    # graph-based segmentation and digits extraction
    # ------------------------
    segmented = seg.GraphBasedSegmentation(image)
    segmented.segment(
                      k=4500
                    , min_size=100
                    , preprocessing=True
                    , gaussian_blur=2.3)

    segmented.generate_image()
    segmented.draw_boxes()
    segmented.extract_digits()

    save_digits(segmented.digits, 'img/digits.png')
    # ------------------------

    output = classifier.classify(segmented.digits)
    print(output)



def train(augmentation, splits, batch_size, epochs, lr, num_workers, device):

    # creating a new classifier
    # ------------------------
    classifier = cnn.CNN(
                      data_augmentation=augmentation
                    , device=device)
    # ------------------------

    print("\n\nDataset preparation ...\n")

    basedir = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = os.path.join(basedir, 'data/')
    print("Dataset folder: {}".format(dataset_folder))

    # preparing training and validation dataset
    # ------------------------
    training_validation_set = dataset.MNIST(
                      folder=dataset_folder
                    , train=True
                    , download_dataset=True
                    , empty=False
                    )
    # ------------------------

    # preparing test dataset
    # ------------------------
    test_set = dataset.MNIST(
                  folder=dataset_folder
                , train=False
                , download_dataset=True
                , empty=False
                )
    # ------------------------

    # splitting dataset into training and validation set
    # ------------------------
    training_set, validation_set = training_validation_set.splits(
                                                  proportions=splits
                                                , shuffle=True
                                                )
    # ------------------------
  
    # setting preprocessing operations if enabled
    # ------------------------
    training_set.set_preprocess(classifier.preprocess)
    # ------------------------


    # print some statistics
    # ------------------------
    print("\n\nStatistics: training set\n")
    training_set.statistics()
    print("\n\nStatistics: validation set\n")
    validation_set.statistics()
    print("\n\nStatistics: test set\n")
    test_set.statistics()
    # ------------------------

    # defining model path (in which models will be saved)
    # ------------------------
    model_path = os.path.join(basedir, 'models/')
    # ------------------------


    # training the classifier
    # ------------------------
    print("\n\nTraining phase...\n")
    classifier.train_cnn(
                  training_set=training_set
                , validation_set=validation_set
                , batch_size=batch_size
                , lr=lr
                , epochs=epochs
                , num_workers=num_workers
                , model_path=model_path
                )
    # ------------------------
    
    # computing the performance of the final model in the prepared data splits
    # ------------------------
    print("\n\nValidation phase...\n")

    # load the best classifier model
    model_name = '{}CNN-batch_size{}-lr{}-epochs{}{}.pth'.format\
                    (model_path, batch_size, lr, epochs, '-a' if augmentation else '')
    classifier.load(model_name)

    training_acc = classifier.eval_cnn(training_set)
    validation_acc = classifier.eval_cnn(validation_set)
    test_acc = classifier.eval_cnn(test_set)

    print("\n\Accuracies\n")

    print("training set:\t{:.2f}".format(training_acc))
    print("validation set:\t{:.2f}".format(validation_acc))
    print("test set:\t{:.2f}".format(test_acc))
    # ------------------------

    print("\n\nModel path: {}\n", os.path.join(model_path, model_name))



def main():
    args = main_args_parser()

    if args.mode == "classify":
        classify(args.webcam, args.folder, args.device)

    elif args.mode == "train":
        train(args.augmentation, args.splits, args.batch_size, args.epochs, args.learning_rate, args.num_workers, args.device)

    else:
        raise ValueError("Execution mode must be either (classify) or (train)")




if __name__ == "__main__":
    main()