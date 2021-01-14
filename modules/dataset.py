""" modules.dataset.py
Summary
-------
This module contains the MNIST dataset class and related functions. 

Classes
-------
MNIST
    implements the MNIST dataset class

Functions
---------
is_downloadable()
    checks if a url contains downloadable resource

download()
    downloads a `.gz` file from a url and saves it to a folder with given filename

store_file_to_tensor()
    reads an `IDX` file and stores its content into a `torch.Tensor`


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

import os
import sys
import gzip
import torch
import struct
import shutil
import requests
import torchvision




def is_downloadable(url: str) -> bool:
    """ Checks if the url contains a downloadable resource.

    Parameters
    ----------
    url: str
        url of the source to be downloaded

    Returns
    -------
    is_downloadable: bool
        returns True if the source has application/x-gzip as content-type 
    """
    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get('content-type')
    if 'application/x-gzip' in content_type.lower():
        return True
    return False



def download(
    url: str, 
    folder: str, 
    filename: str
    ) -> None:
    """ Downloads a .gz file from the provided url and saves it to a folder.

    Parameters
    ----------
    url: str
        url of the source to be downloaded

    folder: str
        folder in which the file will be downloaded

    filename: str
        name of the file that will be stored in folder
    """
    
    if is_downloadable(url):
        if url.find('.'):
            compressed_filename = filename + '.' + url.rsplit('.', 1)[-1] # filename.gz

        # get the file content
        r = requests.get(url, stream=True)

        if not os.path.exists(folder):   
            os.makedirs(folder)          

        compressed_file_path = os.path.join(folder, compressed_filename)
        file_path = os.path.join(folder, filename)

        if not os.path.exists(compressed_file_path): 
            print("Downloading {} ...".format(compressed_file_path))
            
            with open(compressed_file_path, 'wb') as f:
                f.write(r.raw.data)

        if not os.path.exists(file_path): 
            print("Extracting {} ...".format(file_path))
            
            with gzip.open(compressed_file_path, 'rb') as f_in: # uncompress
                with open(file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)



def store_file_to_tensor(file_path: str) -> torch.tensor:
    """ Reads the file indicated by file_path and stores its content into a `torch.tensor`.

    Parameters
    ----------
    file_path: str
        file path to the file to be read
    
    Returns
    -------
    dataset: torch.tensor
        PyTorch tensor containing the dataset (images or labels) depending on the input file
    """
    
    with open(file_path, 'rb') as f:   
        
        magic_number_list = [byte for byte in f.read(4)] 
        dimensions_list_32bit = [f.read(4) for _ in range(magic_number_list[3])]
        dimensions = [struct.unpack('>I', dimension)[0] for dimension in dimensions_list_32bit]
        
        encoding = {b'\x08':['B',1,torch.uint8], 
                    b'\x09':['b',1,torch.int8], 
                    b'\x0B':['h',2,torch.short], 
                    b'\x0C':['i',4,torch.int32], 
                    b'\x0D':['f',4,torch.float32], 
                    b'\x0E':['d',8,torch.float64]}

        e_format = ">" + encoding[magic_number_list[2].to_bytes(1, byteorder='big')][0]
        n_bytes = encoding[magic_number_list[2].to_bytes(1, byteorder='big')][1]
        d_type = encoding[magic_number_list[2].to_bytes(1, byteorder='big')][2]

        #region - images
        if len(dimensions) == 3:
           
            print('Loading {} ...'.format(file_path))
            
            dataset = torch.tensor([[[struct.unpack(e_format, f.read(n_bytes))[0] 
                                    for _ in range(dimensions[2])] 
                                    for _ in range(dimensions[1])] 
                                    for _ in range(dimensions[0])], 
                                    dtype=d_type)

            print('{} loaded!'.format(file_path))
        #endregion

        #region - labels
        elif len(dimensions) == 1:
        
            print('Loading {} ...'.format(file_path))

            dataset = torch.tensor([struct.unpack(e_format, f.read(n_bytes))[0]
                                    for _ in range(dimensions[0])], 
                                    dtype=d_type)

            print('{} loaded!'.format(file_path))
        #endregion

        else: 
            raise ValueError("invalid dimensions in the IDX file.")

        return dataset



class MNIST(torch.utils.data.Dataset):
    """ MNIST dataset class.

    Derived from torch.nn.Module.

    Provides operation for:
    * downloading the dataset
    * leaving the dataset empty (necessary for splitting)
    * splitting the dataset into validation and training set
    * get DataLoader
    * get classes distribution (needed in CNN for cross entropy)
    """


    def __init__(
        self, 
        folder: str, 
        train: bool=True, 
        download_dataset: bool=False, 
        empty: bool=False
        ) -> None:
        """ Class constructor.

        * download the dataset (if download_dataset==True) and if it is not already downloaded
        * build the training dataset (train==True) or the test dataset (train==False) 
        * leave the dataset empty if specified (empty==True)
        * stores the data and the labels in the corresponding torch.tensor

        Parameters
        ----------
        folder: str
            folder which contains/will contain the data

        train: bool (default=True)
            (if True) builds the training dataset
            (if False) builds the test dataset

        download_dataset: bool (default=False)
            (if True) download the dataset from source 
            (if False) expect the dataset to be already downloaded in folder

        empty: bool (default=False)
            (if True) data and label tensors left empty
            (if False) data and label tensors are filled with contents from MNIST
        """

        #region - initialization
        if folder is None:
            raise FileNotFoundError("Please specify the data folder")
        if os.path.isfile(folder):
            raise FileNotFoundError("Invalid data path: {}".format(folder))
        if not os.path.exists(folder):
            os.makedirs(folder)     

        self.folder = folder
        self.data = None
        self.labels = None
        self.preprocess = None
        #endregion


        #region - download, storing and loading of dataset
        if not empty:

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

            self.raw_folder = os.path.join(self.folder, "mnist/raw")
            self.processed_folder = os.path.join(self.folder, "mnist/processed")
            
            if download_dataset:
                for name, url in urls.items():
                    download(url, self.raw_folder, name)
                    
            else:
                if not os.path.exists(self.raw_folder) or os.path.isfile(self.raw_folder):
                    raise FileNotFoundError("invalid data path: {}".format(self.raw_folder))

            filepath = os.path.join(self.processed_folder, self.save_file)

            if os.path.exists(filepath) and os.path.isfile(filepath): # *.pt file present
                self.load(filepath)

            else:   # *.pt file not present
                for name, _ in urls.items():
                    filepath = os.path.join(self.raw_folder, name)
                    
                    if "images" in name:
                        self.data = store_file_to_tensor(filepath)
                        # add one dimension to X to give it as input to CNN by forward
                        self.data = torch.unsqueeze(self.data, 1)                    
                        # convert from uint8 to float32 due to runtime problem in conv2d forward phase
                        self.data = self.data.type(torch.FloatTensor)

                    elif "labels" in name:
                        self.labels = store_file_to_tensor(filepath)
                self.save()
        #endregion
            

    def __len__(self) -> int:
        """ Returns the lenght of the dataset.

        Returns
        -------
        length: int 
            dataset dimension (number of samples)
        """
        return len(self.data) if self.data is not None else 0

    
    def __getitem__(
        self, 
        idx: int
        ) -> tuple:
        """ Gets the item of the dataset at index idx.

        If the preprocess had been set (`set_preprocess()`) the item is returned after being preprocessed.

        Parameters
        ----------
        idx: int
            index of the item to be retrieved.
        
        Returns
        -------
        item: tuple
            item at index idx (image, label) 
        """
        img, label = self.data[idx], int(self.labels[idx])

        if self.preprocess is not None:
            img = self.preprocess(img)

        return (img, label)
    

    def save(self) -> None:
        """ Saves the dataset.
        
        The dataset (tuple of torch.tensors) is saved into a file defined by `processed_folder` and `save_file`.
        """
        if not os.path.exists(self.processed_folder):   
            os.makedirs(self.processed_folder)  

        with open(os.path.join(self.processed_folder, self.save_file), 'wb') as f:
            torch.save((self.data, self.labels), f)


    def load(
        self, 
        file_path: str
        ) -> None:
        """ Loads the `.pt` file in the path defined by `file_path` into data and label tensors.
        
        Parameters
        ----------
        file_path: str
            path to the file to be loaded
        """

        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            raise FileNotFoundError("folder not present: {}".format(file_path))

        self.data, self.labels = torch.load(file_path)


    def set_preprocess(
        self, 
        operations: torch.nn.Sequential or torchvision.transforms or torchvision.transforms.Compose
        ) -> None:
        """ Sets a custom preprocess operation to be applied to each sample.

        Parameters
        ----------
        operations: torch.nn.Sequential or torchvision.transforms or torchvision.transforms.Compose
            preprocesses operation to be applied to each sample

        Example
        -------
            torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(30),
                torchvision.transforms.RandomResizedCrop(28, scale=(0.9,1.1)),
                ])
        """
        self.preprocess = operations


    def splits(
        self, 
        proportions: list=[0.7, 0.3], 
        shuffle: bool=True
        ) -> list:
        """ Splits the the dataset according to the given proportions and return several instances of MNIST (depending on `len(proportions)`).

        Parameters
        ----------
        proportions: list (default=[0.7,0.3]) 
            list of proportions for training set and validation set.
        
        shuffle: bool (default=True) 
            (if True) the dataset is randomly shuffled before being split
            (if False) the dataset is not shuffled

        Returns
        -------
        datasets: list
            list of MNIST dataset

        Notes
        -----
        The number of returned datasets depends on the lenght of the input proportions list.  
        For the particular case it is necessary split into 2 datasets, but the implementation of this function can be useful for other cases.
        """
        
        #region - checks
        if len(self.data) == 0 or len(self.labels) == 0:
            raise RuntimeError("empty dataset cannot be splitted. Please fill it first!")

        if not (sum(proportions) == 1. and all([p > 0. for p in proportions])):
            raise ValueError("invalid proportions. They must (1) be greater than zero (2) sum up to 1")
        #endregion
        

        num_splits = len(proportions)
        dataset_size = self.data.shape[0]

        datasets = []
        for i in range(num_splits):
            datasets.append(MNIST(folder=self.folder, empty=True))

        if shuffle:
            permutation = torch.randperm(dataset_size)
        else:
            permutation = torch.arange(dataset_size)
            
        data = self.data[permutation]
        labels = self.labels[permutation]

        start = 0

        #region - splitting
        for i in range(num_splits):
            num_samples =int(proportions[i] * dataset_size)
            end = start + num_samples if i < num_splits - 1 else dataset_size

            datasets[i].data = data[start:end]
            datasets[i].labels = labels[start:end]

            start = end
        #endregion

        return datasets


    def get_loader(
        self, 
        batch_size: int=1, 
        num_workers: int=0, 
        shuffle: bool=True
        ) -> torch.utils.data.DataLoader:
        """ Returns the DataLoader for the current dataset.

        Parameters
        ----------
        batch_size: int (default=1)
            how many samples per batch to load
        num_workers: int (default=0)
            how many subprocesses to use for data loading
            0 means that the data will be loaded in the main process
        shuffle: bool (default=True)
            (if True) data is randomly shuffled at every epoch
        
        Returns
        -------
        data_loader: torch.utils.data.DataLoader
            provides an iterable over the current dataset
        """
        
        #region - checks
        if len(self.data) == 0 or len(self.labels) == 0:
            raise RuntimeError("empty dataset cannot be splitted. Please fill it first!")
        #endregion
        
        data_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, 
                                                    shuffle=shuffle, num_workers=num_workers)
            
        return data_loader
    
    
    def classes_distribution(self) -> None:
        """ Returns the classes distribution.

        Used in cross-entropy for taking care of little unbalancements between population of classes.
        """
        return self.labels.bincount() / len(self.labels)


    def statistics(self) -> None:
        """ Prints some basic statistics of the current dataset. """
        print("N. samples:    \t{0}".format(len(self.data)))
        print("Classes:       \t{0}".format(set(self.labels.numpy())))
        print("Classes distr.: {0}".format([round(i.item(), 4) for i in self.classes_distribution()]))
        print("Data type:     \t{0}".format(type(self.data[0])))
        print("Data dtype:     {0}".format(self.data[0].dtype))
        print("Data shape:    \t{0}\n".format(self.data[0].shape))