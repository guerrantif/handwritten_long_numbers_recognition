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

import numpy as np
import torch
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Any, Callable, Optional, Sequence



class MnistDataset():
    """
    Dataset class for MNIST dataset.
    It provides an easy downloader and the possibility to split the dataset.
    """

    def __init__(
          self
        , path: str='./data/'
        , download: bool=True
        , normalize: bool=True
        ) -> None:
        """
        MnistDataset class constructor.

        Args:
            path        (str): path in which the dataset is present 
                               or in which it will be downloaded if not present yet
            download   (bool): if True and if the dataset is not present yet it will be downloaded 
            normalize  (bool): if True the images will be normalized (both in training set and test set)
        """

        if normalize:
            transform = transforms.Compose([
                              transforms.ToTensor()
                            , transforms.Normalize((0.1307,), (0.3081,))   # mean and std of MNIST dataset
                            ])
        else:
            transform = None

        # data_set inizialization (base for training_set and validation_set)
        self.data_set = torchvision.datasets.MNIST(
                                  root=path
                                , train=True
                                , download=download
                                , transform=transform
                                )
        # test_set initialization
        self.test_set = torchvision.datasets.MNIST(
                                  root=path
                                , train=False
                                , download=False
                                , transform=transform
                                )

        # setting splitting indices to None
        self.training_indices, self.validation_indices = None, None 


    @staticmethod
    def __statistics(
          dataset: torchvision.datasets
        , train: Optional[bool]=True
        ) -> None:
        """
        UTILITY FUNCTION: prints some basic statistics of the MNIST dataset.
        
        Args:
            dataset (torchvision.datasets): training or test set of MNIST dataset
            train (bool): if True the train set statistics are printed, else the test set ones
        """

        if train: print("TRAIN set")
        else: print("TEST set")
        print("N. samples:    \t{0}".format(len(dataset)))
        print("Targets:       \t{0}".format(set(dataset.targets.numpy())))
        print("Target distr.: \t{0}".format(dataset.targets.bincount()/len(dataset)*100))
        image, label = next(iter(dataset))
        print("Data type:     \t{0}".format(type(image)))
        print("Data shape:    \t{0}\n".format(image.shape))


    def statistics(self) -> None:
        """
        Prints some basic statistics of the MNIST dataset.
        """

        MnistDataset.__statistics(self.data_set, True)
        MnistDataset.__statistics(self.test_set, False)


    def show(self)-> None:
        '''
        Shows the first 20 images of MNIST training set.
        '''
        
        plt.figure(figsize=(50,50))
        for i, sample in enumerate(self.data_set, start=1):
            image, label = sample
            plt.subplot(10, 10, i)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.axis('off')
            plt.title(self.data_set.classes[label], fontsize=28)
            if (i >= 20): break
        plt.show()


    def create_splits(
          self
        , proportions: list
        , shuffle: bool=True
        ) -> None:
        """
        Creates the indices for splitting the dataset into training and validation sets by the given proportions.

        Args:
            proportions (list): list with the fractions of data to store in training set and validation set respectively.
            shuffle     (bool): whether to shuffle the dataset indices before splitting
        """

        # check proportions
        if sum(proportions) == 1. and all([p > 0. for p in proportions]) and len(proportions) == 2:
            pass
        else:
            raise ValueError("Invalid proportions: they must (1) be 2 (2) sum up to 1 (3) be all positives.")

        
        # creating data indices for training and validation splits
        length = len(self.data_set)
        indices = np.arange(length)
        split = int(np.floor(proportions[0] * length))

        if shuffle:
            np.random.shuffle(indices)  # in-place operation
        
        self.training_indices, self.validation_indices = indices[:split], indices[split:]


    def get_loader(
          self
        , set: str
        , batch_size: int=1
        , num_workers: int=0
        ) -> DataLoader:
        """
        Returns the DataLoader for the given set {training, validation, test}.

        Args:
            set         (str):  {'training', 'validation', 'test'}, the set of which a DataLoader will be returned.
            batch_size  (int):  how many samples per batch to load.
            shuffle    (bool):  set to True to have the data reshuffled at every epoch.
            num_workers (int):  how many subprocesses to use for data loading. 
                                0 means that the data will be loaded in the main process.
        
        Returns:
            data_loader (DataLoader): provides an iterable over the given dataset.
        """

        if self.training_indices is None and self.validation_indices is None:
            raise ValueError("Create splits first!")

        if set == "training":
            sampler = SubsetRandomSampler(self.training_indices)
            data_loader = DataLoader(
                                  self.data_set
                                , batch_size=batch_size
                                , sampler=sampler
                                , num_workers=num_workers
                                )

        elif set == "validation":
            sampler = SubsetRandomSampler(self.validation_indices)
            data_loader = DataLoader(
                                  self.data_set
                                , batch_size=batch_size
                                , sampler=sampler
                                , num_workers=num_workers
                                )

        elif set == "test":
            data_loader = DataLoader(
                                  self.test_set
                                , batch_size=batch_size
                                , num_workers=num_workers
                                )

        else:
            raise ValueError("Invalid set: {'training', 'validation', 'test'}.")

            
        return data_loader