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


import torch
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import random
from math import floor
from typing import Any, Callable, TypeVar, Generic, Sequence, List, Optional
import matplotlib.pyplot as plt



class MNIST_dataset():
    """
    Class handler for MNIST dataset.
    """
    training_set: datasets.MNIST
    test_set: datasets.MNIST

    def __init__(
        self, 
        path: str = 'data/',
        train_transform: Optional[Callable],
        test_transform: Optional[Callable],
        ) -> None:
        """
        Class constructor. 
        The constructor will download the dataset, if not present, in the path directory.
        Three class members will be initialized:
            - training_set (torchvision.datasets.MNIST)     : is the training set
            - test_set (torchvision.datasets.MNIST)         : is the test set

        Args:
            path (str): the path in which the dataset is present or in which it will be downloaded
                        if not present
            train_transform (callable, optional): a function/transform that takes a train image
                        and returns a transformed version
            test_transform (callable, optional): a function/transform that takes a test image
                        and returns a transformed version
        """

        self.dataset = datasets.MNIST(root=path, train=True, download=True, transform=train_transform)
        self.test_set = datasets.MNIST(root=path, train=False, transform=test_transform)

        self.dataset_len = len(self.dataset)
        self.test_set_len = len(self.test_set)

    
    # def split(
    #     self, 
    #     proportions: list = [0.7, 0.15],
    #     shuffle: bool = True,
    #     balanced: bool = True,
    #     ) -> list():
    #     """
    #     The dataset splits will be created. In the case of the MNIST dataset, only training set and validation set
    #     must be splitted, since the test set comes already separated.

    #     Args:
    #         proportions (list): list of two floats containing the proportions in which the training set and the 
    #                         validation test will be respectively splitted. They must sum up to one and be both 
    #                         greater than zero.
    #         shuffle (bool): if True the training and validation datasets will be shuffled before being partitioned.
    #         balanced (bool): if True the classes will be splitted in a balanced way among training and validation sets.

    #     Returns:
    #         training_loader (torch.utils.data.DataLoader): training set iterator
    #         validation_loader (torch.utils.data.DataLoader): validation set iterator
    #         test_loader (torch.utils.data.DataLoader): test set iterator
    #     """

    #     # checking argument
    #     if (sum(proportions) != 1. or any(p <= 0. for p in proportions)) and len(proportions) != 2:
    #         raise ValueError("Invalid split proportions: they must be (1) two floats (2) sum up to 1 (3) be greater than 0.")


    #     if balanced:    # the classes will be balanced among training and validation sets

    #         # indices of the images for each class
    #         indices_per_class = [[i for i in range(self.training_set_len) if int(self.labels[i]) == j] for j in set(self.labels)]
    #         training_indices = list()
    #         validation_indices = list()

    #         for class_indices in indices_per_class:
                
    #             indices = class_indices
    #             if shuffle:
    #                 random.shuffle(indices)
    #             split = floor(proportions[0]*len(indices))
    #             training_indices.append(indices[:split])
    #             validation_indices.append(indices[split:])
            
    #         training_indices = [j for i in training_indices for j in i]
    #         validation_indices = [j for i in validation_indices for j in i]
            
    #         if shuffle:
    #             random.shuffle(training_indices)
    #             random.shuffle(validation_indices)

    #     return training_indices, validation_indices


        # indices = list(range(len(self.training_set)))
        # split = int

        


    def __len__(self) -> int:
        """
        Overloaded version of dataset.MNIST.__len__().

        Returns:
            len (int): number of samples of the entire dataset
        """
        return self.training_set_len + self.test_set_len

    
    def trainLoader(self):
        pass


    def testLoader(self)




if __name__ == "__main__":
    
    dataset = MNIST_dataset()
    print(dataset.labels)
    # training_indices, validation_indices = dataset.split(proportions=[0.7, 0.3],
    #                                                      shuffle=True,
    #                                                      balanced=True)
    # print(training_indices)
    # print(validation_indices)