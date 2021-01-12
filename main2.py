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
    
    args = parser.parse_args()
    return args



def classify(webcam, image_path, device):

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
    classifier.load('models/CNN-batch_size150-lr0.001-epochs40-a.pth')
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
    # ------------------------

    fig = plt.figure(figsize=(30,15))
    for i in range(len(segmented.digits)):
        image = segmented.digits[i][0]
        sp = fig.add_subplot(3, len(segmented.digits), i+1)
        plt.axis('off')
        plt.imshow(image, cmap='gray')
    plt.savefig('prva.png')

    output = classifier.classify(segmented.digits)
    print(output)



def main():
    args = main_args_parser()

    if args.mode == "classify":
        classify(args.webcam, args.folder, args.device)




if __name__ == "__main__":
    main()