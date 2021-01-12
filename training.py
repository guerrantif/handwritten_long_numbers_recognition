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

from lib.utils import training_parse_args
from lib.cnn import CNN
from lib.dataset import MNIST
import os



# entry point
if __name__ == "__main__":

    args = training_parse_args()

    # creating a new classifier
    # ------------------------
    classifier = CNN(
                      data_augmentation=args.data_augmentation
                    , device=args.device)
    # ------------------------

    for k,v in vars(args).items():
      print(k,v)

    print("\n\nDataset preparation ...\n")
    # preparing training and validation dataset
    # ------------------------
    training_validation_set = MNIST(
                      folder=args.dataset_folder
                    , train=True
                    , download_dataset=True
                    , empty=False
                    )
    # ------------------------

    # preparing test dataset
    # ------------------------
    test_set = MNIST(
                  folder=args.dataset_folder
                , train=False
                , download_dataset=True
                , empty=False
                )
    # ------------------------

    # splitting dataset into training and validation set
    # ------------------------
    training_set, validation_set = training_validation_set.splits(
                                                  proportions=args.splits
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
    basedir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(basedir, 'models/')
    # ------------------------


    # training the classifier
    # ------------------------
    print("\n\nTraining phase...\n")
    classifier.train_cnn(
                  training_set=training_set
                , validation_set=validation_set
                , batch_size=args.batch_size
                , lr=args.lr
                , epochs=args.epochs
                , num_workers=args.workers
                , model_path=model_path
                )
    # ------------------------
    
    # computing the performance of the final model in the prepared data splits
    # ------------------------
    print("\n\nValidation phase...\n")

    # load the best classifier model
    classifier.load('{}CNN-batch_size{}-lr{}-epochs{}{}.pth'
                    .format(model_path, args.batch_size, args.lr, args.epochs, '-a' if args.data_augmentation else ''))

    training_acc = classifier.eval_cnn(training_set)
    validation_acc = classifier.eval_cnn(validation_set)
    test_acc = classifier.eval_cnn(test_set)

    print("training set:\tacc:{:.2f}".format(training_acc))
    print("validation set:\tacc:{:.2f}".format(validation_acc))
    print("test set:\tacc:{:.2f}".format(test_acc))
    # ------------------------