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

import utils
import torch
import torchvision
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
        if os.path.isfile(folder):
            raise FileNotFoundError("Invalid data path: {}".format(folder))
        if not os.path.exists(folder):
            os.makedirs(folder)     


        self.folder = folder
        # ------------------------

        # dataset attributes
        # ------------------------
        self.data = None
        self.labels = None
        self.preprocess = None
        # ------------------------


        # downloading, storing and loading dataset
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
            self.raw_folder = os.path.join(self.folder, "mnist/raw")
            self.processed_folder = os.path.join(self.folder, "mnist/processed")
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
            filepath = os.path.join(self.processed_folder, self.save_file)
            if os.path.exists(filepath) and os.path.isfile(filepath): # *.pt file present
                self.load(filepath)
            else:   # *.pt file not present
                for name, _ in urls.items():
                    filepath = os.path.join(self.raw_folder, name)
                    
                    if "images" in name:
                        self.data = utils.store_file_to_tensor(filepath)
                        # add one dimension to X to give it as input to CNN by forward
                        self.data = torch.unsqueeze(self.data, 1)                    
                        # convert from uint8 to float32 due to runtime problem in conv2d forward phase
                        self.data = self.data.type(torch.FloatTensor)

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

        if self.preprocess is not None:
            img = self.preprocess(img)

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


    def set_preprocess(
          self
        , operations: torchvision.transforms or torch.nn.Sequential or torchvision.transforms.Compose
        ) -> None:
        """
        Set a custom preprocess operation to be applied to each sample.
        """
        self.preprocess = operations


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
        # check not empty dataset
        # ------------------------
        if len(self.data) == 0 or len(self.labels) == 0:
            raise ValueError("Empty dataset: cannot be splitted. Please fill it first!")
        # ------------------------

        # check proportions
        # ------------------------
        if not (sum(proportions) == 1. and all([p > 0. for p in proportions])):
            raise ValueError("Invalid proportions: they must (1) be 2 (2) sum up to 1")
        # ------------------------

        num_splits = len(proportions)
        dataset_size = self.data.shape[0]

        # creating a list of MNIST objects
        # ------------------------
        datasets = []
        for i in range(num_splits):
            datasets.append(MNIST(folder=self.folder, empty=True))
        # ------------------------

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


    def get_loader(
          self
        , batch_size: int=1
        , num_workers: int=0
        , shuffle: bool=True
        ) -> torch.utils.data.DataLoader:
        """
        Returns the DataLoader for the given set {training, validation, test}.

        Args:
            batch_size  (int):  how many samples per batch to load.
            shuffle    (bool):  set to True to have the data reshuffled at every epoch.
            num_workers (int):  how many subprocesses to use for data loading. 
                                0 means that the data will be loaded in the main process.
        
        Returns:
            data_loader (torch.utils.data.DataLoader): provides an iterable over the given dataset.
        """
        # check not empty dataset
        # ------------------------
        if len(self.data) == 0 or len(self.labels) == 0:
            raise ValueError("Empty dataset: cannot be splitted. Please fill it first!")
        # ------------------------
        
        data_loader = torch.utils.data.DataLoader(
                                      self
                                    , batch_size=batch_size
                                    , shuffle=True
                                    , num_workers=num_workers
                                    )
            
        return data_loader
    
    
    def classes_distribution(self) -> None:
        """
        Return the classes distribution percentage.
        Used in cross-entropy for taking care of little unbalancments.
        """
        return self.labels.bincount() / len(self.labels)


    def statistics(self) -> None:
        """
        Print some basic statistics of the current dataset.
        """
        print("N. samples:    \t{0}".format(len(self.data)))
        print("Classes:       \t{0}".format(set(self.labels.numpy())))
        print("Classes distr.: {0}".format([round(i.item(), 4) for i in self.classes_distribution()]))
        print("Data type:     \t{0}".format(type(self.data[0])))
        print("Data dtype:     {0}".format(self.data[0].dtype))
        print("Data shape:    \t{0}\n".format(self.data[0].shape))