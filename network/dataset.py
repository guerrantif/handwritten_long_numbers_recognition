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

import utils
import torch
import os



class MNIST(torch.utils.data.Dataset):

    def __init__(
          self
        , folder: str
        , train: bool=True
        , download: bool=False
        , empty: bool=False
        ) -> None:
        """
        Class constructor.

        Args:
            folder      (str): folder in which contains/will contain the data
            train      (bool): if True the training dataset is built, otherwise the test dataset
            download   (bool): if True the dataset will be downloaded (default = True)
            empty      (bool): if True the tensors will be left empty (default = False)
        """

        # user folder check
        # ------------------------
        if folder is None:
            raise FileNotFoundError("Please specify the data folder")
        if not os.path.exists(folder) or os.path.isfile(folder):
            raise FileNotFoundError("Invalid data path: {}".format(folder))

        self.folder = folder
        # ------------------------

        # dataset attributes
        # ------------------------
        self.data = None
        self.labels = None
        # ------------------------


        # splitting dataset
        # ------------------------
        if not empty:

            # file in which to save the tensors
            # ------------------------
            if train:
                urls = {
                    'training-images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
                    , 'training-labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
                }
                self.save_file = 'training.pt'
            else:
                urls = {
                    'test-images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
                    , 'test-labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
                }
                self.save_file = 'test.pt'
            # ------------------------

            # folders in which to save the raw dataset
            # ------------------------
            self.raw_folder = os.path.join(self.folder, "data/raw")
            self.processed_folder = os.path.join(self.folder, "data/processed")
            # ------------------------

            # dataset download
            # ------------------------
            if download:
                for name, url in urls.items():
                    utils.download(url, self.raw_folder, name)
            # ------------------------
            
            # dataset folder check
            # ------------------------
            else:   # not download
                if not os.path.exists(self.raw_folder) or os.path.isfile(self.raw_folder):
                    raise FileNotFoundError("Invalid data path: {}".format(self.raw_folder))
            # ------------------------

            # data storing
            # ------------------------
            for name, _ in urls.items():
                filepath = os.path.join(self.raw_folder, name)
                if "images" in name:
                    self.data = utils.store_file_to_tensor(filepath)
                elif "labels" in name:
                    self.labels = utils.store_file_to_tensor(filepath)
            self.save()
            # ------------------------
        # ------------------------
            
    
    def __len__(self) -> int:
        """
        Return the lenght of the dataset.

        Returns:
            length of the dataset (int)
        """
        return len(self.data) if self.data is not None else 0

    
    def __getitem__(
          self
        , idx: int
        ) -> tuple:
        """
        Retrieve the item of the dataset at index idx.

        Args:
            idx (int): index of the item to be retrieved.
        
        Returns:
            tuple: (image, label) 
        """
        img, label = self.data[idx], int(self.labels[idx])

        return (img, label)
    


    def save(self) -> None:
        """
        Save the dataset (tuple of torch.tensors) into a file defined by self.processed_folder and self.save_file.
        """
        if not os.path.exists(self.processed_folder):   
            os.makedirs(self.processed_folder)  

        # saving the training set into the correct folder
        with open(os.path.join(self.processed_folder, self.save_file), 'wb') as f:
            torch.save((self.data, self.labels), f)


    def load(
          self
        , path: str
        ) -> None:
        """
        Load the file .pt in the path defined by self.processed_folder and self.save_file.
        """

        if not os.path.exists(path):
            raise FileNotFoundError("Folder not present: {}".format(path))

        self.data, self.labels = torch.load(path)

    
    def splits(
          self
        , proportions: list=[0.8, 0.2]
        , shuffle: bool=True
        ) -> None:
        """
        Split the the dataset according to the given proportions and return two instances of MNIST, training and validation.

        Args:
            proportions (list): (default=[0.8,0.2]) list of proportions for training set and validation set.
            shuffle (bool): (default=True) whether to shuffle the dataset or not
        """

        # check proportions
        # ------------------------
        if not (sum(proportions) == 1. and all([p > 0. for p in proportions])): #and len(proportions) == 2:
            raise ValueError("Invalid proportions: they must (1) be 2 (2) sum up to 1") # (3) be all positives.")
        # ------------------------

        num_splits = len(proportions)
        dataset_size = self.data.shape[0]

        # creating a list of MNIST objects
        # ------------------------
        datasets = []
        for i in range(num_splits):
            datasets.append(MNIST(folder=self.folder, empty=True))
        # ------------------------
        # return datasets


        if shuffle:
            # create a random permutation of the indices
            # ------------------------
            permutation = torch.randperm(dataset_size)
            # ------------------------
        else:
            # leave the indices unchanged
            # ------------------------
            permutation = torch.arange(dataset_size)
            # ------------------------

        data = self.data[permutation]
        labels = self.labels[permutation]


        # split data and labels into the datasets according to partitions
        # ------------------------
        start = 0

        for i in range(num_splits):
            num_samples =int(proportions[i] * dataset_size)
            end = start + num_samples if i < num_splits - 1 else dataset_size

            datasets[i].data = data[start:end]
            datasets[i].labels = labels[start:end]

            start = end
        # ------------------------

        return datasets