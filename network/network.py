"""
Copyright January 2021 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

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

import utils
import cnn
import dataset



# entry point
if __name__ == "__main__":

    args = utils.parse_command_line_arguments()

    print('\n')
    for k, v in args.__dict__.items():
        print(k + '=' + str(v))


    if args.mode == 'train':

        # creating a new classifier
        # ------------------------
        classifier = cnn.CNN(
                          data_augmentation=args.preprocess
                        , device=args.device)
        # ------------------------

        print("\n\nDATASET PREPARATION\n")
        # preparing training and validation dataset
        # ------------------------
        training_validation_set = dataset.MNIST(
                          folder=args.dataset_folder
                        , train=True
                        , download=True
                        , empty=False
                        )
        # ------------------------

        # preparing test dataset
        # ------------------------
        test_set = dataset.MNIST(
                          folder=args.dataset_folder
                        , train=False
                        , download=True
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
        # validation_set.set_preprocess(classifier.preprocess)
        # ------------------------


        # print some statistics
        # ------------------------
        print("\n\nTRAINING SET STATISTICS\n")
        training_set.statistics()
        print("\n\nVALIDATION SET STATISTICS\n")
        validation_set.statistics()
        print("\n\nTEST SET STATISTICS\n")
        test_set.statistics()
        # ------------------------

        # training the classifier
        # ------------------------
        print("\n\nTRAINING PHASE\n")
        classifier.train_cnn(
                      training_set=training_set
                    , validation_set=validation_set
                    , batch_size=args.batch_size
                    , lr=args.lr
                    , epochs=args.epochs
                    , num_workers=args.workers
                    , model_path=args.model_path
                    )
        # ------------------------
        
        # computing the performance of the final model in the prepared data splits
        # ------------------------
        print("\n\nVALIDATION PHASE\n")
        training_acc = classifier.eval_cnn(training_set)
        validation_acc = classifier.eval_cnn(validation_set)
        test_acc = classifier.eval_cnn(test_set)

        print("training set:\tacc:{:.2f}".format(training_acc))
        print("validation set:\tacc:{:.2f}".format(validation_acc))
        print("test set:\tacc:{:.2f}".format(test_acc))
        # ------------------------




    # elif args.mode == 'eval':
    #     print("Evaluating the classifier...")

    #     # creating a new classifier
    #     _classifier = cnn.CNN(
    #                       model=args.model
    #                     , device=args.device
    #                     )

    #     # loading the classifier
    #     _classifier.load(args.model_name)

        # image preprocessing