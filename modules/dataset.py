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

import struct
import requests
import os
import gzip
import shutil
import torch
import torchvision
import os
import sys




def is_downloadable(url: str) -> bool:
    """
    Check if the url contains a downloadable resource.

    Args:
        url     (str): url of the source to be downloaded

    Returns:
        is_downloadable (bool): True if the source has application/x-gzip as content-type 
    """
    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get('content-type')
    if 'application/x-gzip' in content_type.lower():
        return True
    return False


def download(
      url: str
    , folder: str
    , filename: str
    ) -> None:
    """
    Download a .gz file from the provided url and saves it to a folder.

    Args:
        url         (str): url of the source to be downloaded
        folder      (str): folder in which the file will be saved
        filename    (str): name to use as filename
    """
    if is_downloadable(url):
        # compressed filename (filename.gz)
        if url.find('.'):
            compressed_filename = filename + '.' + url.rsplit('.', 1)[-1]

        # get the file content
        r = requests.get(url, stream=True)

        # check for the existence of directory
        if not os.path.exists(folder):   
            # creation of the folder if it doesn't exist
            os.makedirs(folder)          

        compressed_file_path = os.path.join(folder, compressed_filename)
        file_path = os.path.join(folder, filename)

        if not os.path.exists(compressed_file_path): # if no downloads present

            print("Downloading {} ...".format(compressed_file_path))
            
            # write the files
            with open(compressed_file_path, 'wb') as f:
                f.write(r.raw.data)


        if not os.path.exists(file_path):   # if uncompressed file is present 

            print("Extracting {} ...".format(file_path))
            
            # open the compressed file
            with gzip.open(compressed_file_path, 'rb') as f_in:
                # open the uncompressed file to be filled
                with open(file_path, 'wb') as f_out:
                    # fill the uncompressed file
                    shutil.copyfileobj(f_in, f_out)


def store_file_to_tensor(file_path: str) -> torch.tensor:
    """
    Reads the file indicated by file_path and stores its content into a PyTorch tensor.

    Args:
        file_path   (str): file path to the file to be read
    
    Returns:
        dataset (torch.tensor): PyTorch tensor containing the dataset (images or label),
                                depending on the input file
    """
    
    # open the file for reading in binary mode 'rb'
    with open(file_path, 'rb') as f:     
        # magic number list   
        m_numb_list = [byte for byte in f.read(4)] 
        # dimensions list  
        d_list_32bit = [f.read(4) for _ in range(m_numb_list[3])]
        dimensions = [struct.unpack('>I', dimension)[0] for dimension in d_list_32bit]
        
        encoding = {
                      b'\x08':['B',1,torch.uint8]
                    , b'\x09':['b',1,torch.int8]
                    , b'\x0B':['h',2,torch.short]
                    , b'\x0C':['i',4,torch.int32]
                    , b'\x0D':['f',4,torch.float32]
                    , b'\x0E':['d',8,torch.float64]
                    }

        e_format = ">" + encoding[m_numb_list[2].to_bytes(1, byteorder='big')][0]
        n_bytes = encoding[m_numb_list[2].to_bytes(1, byteorder='big')][1]
        d_type = encoding[m_numb_list[2].to_bytes(1, byteorder='big')][2]


        if len(dimensions) == 3:    # images
           
            print('Loading {} ...'.format(file_path))
            
            dataset = torch.tensor(
                [
                    [
                        [struct.unpack(e_format, f.read(n_bytes))[0] 
                        for _ in range(dimensions[2])] 
                    for _ in range(dimensions[1])] 
                for _ in range(dimensions[0])]
                , dtype=d_type
            )

            print('{} loaded!'.format(file_path))
        

        elif len(dimensions) == 1:  # labels
        
            print('Loading {} ...'.format(file_path))

            dataset = torch.tensor(
                [struct.unpack(e_format, f.read(n_bytes))[0]
                for _ in range(dimensions[0])]
                , dtype=d_type
            )

            print('{} loaded!'.format(file_path))
        

        else:   # wrong dimensions
            raise ValueError("Invalid dimensions in the IDX file!")

        
        return dataset



class MNIST(torch.utils.data.Dataset):


    def __init__(
          self
        , folder: str
        , train: bool=True
        , download_dataset: bool=False
        , empty: bool=False
        ) -> None:
        """
        Class constructor.

        Args:
            folder      (str): folder in which contains/will contain the data
            train      (bool): if True the training dataset is built, otherwise the test dataset
            download_dataset   (bool): if True the dataset will be downloaded (default = True)
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
            if download_dataset:
                for name, url in urls.items():
                    download(url, self.raw_folder, name)
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
                        self.data = store_file_to_tensor(filepath)
                        # add one dimension to X to give it as input to CNN by forward
                        self.data = torch.unsqueeze(self.data, 1)                    
                        # convert from uint8 to float32 due to runtime problem in conv2d forward phase
                        self.data = self.data.type(torch.FloatTensor)

                    elif "labels" in name:
                        self.labels = store_file_to_tensor(filepath)
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