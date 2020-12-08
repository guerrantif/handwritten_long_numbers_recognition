"""
Copyright December 2020 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

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

import cnn
import dataset
import argparse

def parse_command_line_arguments():
    """Parse command line arguments, checking their values."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', choices=['train', 'classify'],
                        help='train the classify or classify image')
    parser.add_argument('--dataset_path', type=str, default='./../data/',
                        help='dataset path')
    parser.add_argument('--image_path', type=str, default='./../img/',
                        help='image path')
    parser.add_argument('--model_name', type=str, default=None, 
                        help='model to load from memory (if mode == eval)')
    parser.add_argument('--model_path', type=str, default='./../models/',
                        help='image path')
    parser.add_argument('--splits', type=str, default='0.7-0.3',
                        help='fraction of data to be used in training and validation set (default: 0.7-0.3)')
    parser.add_argument('--model', type=str, default='model1', choices=['model1', 'model2'],
                        help='model of the CNN (default: model1)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'],
                        help='optimizer (default: adam)')
    parser.add_argument('--workers', type=int, default=3,
                        help='number of working units used to load the data (default: 3)')
    parser.add_argument('--device', default='cpu', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parsed_arguments = parser.parse_args()

    # converting split fraction string to a list of floating point values ('0.7-0.3' => [0.7, 0.3])
    splits_string = str(parsed_arguments.splits)
    fractions_string = splits_string.split('-')
    if len(fractions_string) != 2:
        raise ValueError("Invalid split fractions were provided. Required format (example): 0.7-0.3")
    else:
        splits = []
        frac_sum = 0.
        for fraction in fractions_string:
            try:
                splits.append(float(fraction))
                frac_sum += splits[-1]
            except ValueError:
                raise ValueError("Invalid split fractions were provided. Required format (example): 0.7-0.3")
        if frac_sum != 1.0:
            raise ValueError("Invalid split fractions were provided. They must sum to 1.")

    # updating the 'splits' argument
    parsed_arguments.splits = splits

    # checking present of model name if mode == eval
    if parsed_arguments.mode == 'eval' and parsed_arguments.model_name == None:
        raise ValueError("Model name must be provided if mode == 'eval'.")


    return parsed_arguments


# entry point
if __name__ == "__main__":
    args = parse_command_line_arguments()

    for k, v in args.__dict__.items():
        print(k + '=' + str(v))

    if args.mode == 'train':
        print("Training the classifier")

        # creating a new classifier
        _classifier = cnn.CNN(
                          model=args.model
                        , device=args.device
                        )

        # preparing dataset
        _data_set = dataset.MnistDataset(
                          path=args.dataset_path
                        , download=True
                        , normalize=True
                        )

        # print some statistics
        _data_set.statistics()

        # splitting dataset into training and validation set
        _data_set.create_splits(
                                                          proportions=args.splits
                                                        , shuffle=True
                                                        )

        # getting data loaders of datasets
        _training_loader = _data_set.get_loader(
                              set='training'
                            , batch_size=args.batch_size
                            , num_workers=args.workers
                            )
        
        _validation_loader = _data_set.get_loader(
                              set='validation'
                            , batch_size=args.batch_size
                            , num_workers=args.workers
                            )
        
        _test_loader = _data_set.get_loader(
                              set='test'
                            , batch_size=args.batch_size
                            , num_workers=args.workers
                            )

        # training the classifier
        _classifier.train_cnn(
                      training_set=_training_loader
                    , validation_set=_validation_loader
                    , optimizer_mode=args.optim
                    , lr=args.lr
                    , epochs=args.epochs
                    , model_path=args.model_path
                    )
        
        # computing the performance of the final model in the prepared data splits
        print("Evaluating the classifier...")
        _train_acc = _classifier.eval_cnn(_train_set)
        _val_acc = _classifier.eval_cnn(_val_set)
        _test_acc = _classifier.eval_cnn(_test_set)

        print("training set:\tacc:{:.2f}".format(_train_acc))
        print("validation set:\tacc:{:.2f}".format(_val_acc))
        print("test set:\tacc:{:.2f}".format(_test_acc))

    elif args.mode == 'eval':
        print("Evaluating the classifier...")

        # creating a new classifier
        _classifier = cnn.CNN(
                          model=args.model
                        , device=args.device
                        )

        # loading the classifier
        _classifier.load(args.model_name)

        # image preprocessing