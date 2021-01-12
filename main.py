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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('image_path'
                        , type=str
                        , help='relative path to the image to recognize')
    parsed_arguments = parser.parse_args()
    image = parsed_arguments.image_path

    # creating a new classifier
    # ------------------------
    classifier = cnn.CNN(device='cpu')
    # ------------------------

    classifier.load('models/CNN-batch_size150-lr0.001-epochs40-a.pth')

    segmented = seg.GraphBasedSegmentation(image)
    segmented.segment(
                    k=4500
                    , min_size=100
                    , preprocessing=True
                    , gaussian_blur=2.3)

    segmented.generate_image()
    segmented.draw_boxes()
    segmented.extract_digits()

    fig = plt.figure(figsize=(30,15))
    for i in range(len(segmented.digits)):
        image = segmented.digits[i][0]
        sp = fig.add_subplot(3, len(segmented.digits), i+1)
        plt.axis('off')
        plt.imshow(image, cmap='gray')
    plt.savefig('prva.png')

    output = classifier.classify(segmented.digits)
    print(output)

