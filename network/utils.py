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


import requests
import os
import gzip
import shutil
import torch.tensor
import struct
import argparse



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


def training_parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for training phase.
    
    Returns:
        parsed_arguments (argparse.Namespace): populated Namespace: the arguments passed via 
                                            command line are converted to objects and assigned 
                                            as attributes of the namespace
    """

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-a'
                        , '--data_augmentation'
                        , action='store_true'
                        , help='data augmentation preprocessing is applied')

    parser.add_argument('--dataset_folder'
                        , type=str
                        , default='./../data/'
                        , help='(default=\'./../data/\') folder where to save the dataset or from where to load it (if mode == train)')

    parser.add_argument('--splits'
                        , type=str
                        , default='0.8-0.2'
                        , help='(default: 0.8-0.2) fraction of data to be used in training and validation set')

    parser.add_argument('--batch_size'
                        , type=int
                        , default=64
                        , help='(default: 64) mini-batch size')

    parser.add_argument('--epochs'
                        , type=int
                        , default=10
                        , help='(default: 10) number of training epochs')

    parser.add_argument('--lr'
                        , type=float
                        , default=0.001
                        , help='(default: 0.001) learning rate')

    parser.add_argument('--workers'
                        , type=int
                        , default=3
                        , help='(default: 3) number of working units used to load the data')

    parser.add_argument('--device'
                        , default='cpu'
                        , type=str
                        , help='(default: \'cpu\') device to be used for computations {cpu, cuda:0, cuda:1, ...}')

    parsed_arguments = parser.parse_args()


    # converting split fraction string to a list of floating point values ('0.8-0.2' => [0.8, 0.2])
    # ------------------------
    splits_string = str(parsed_arguments.splits)
    fractions_string = splits_string.split('-')
    if len(fractions_string) != 2:
        raise ValueError("Invalid split fractions were provided. Required format (example): 0.8-0.2")
    else:
        splits = []
        frac_sum = 0.
        for fraction in fractions_string:
            try:
                splits.append(float(fraction))
                frac_sum += splits[-1]
            except ValueError:
                raise ValueError("Invalid split fractions were provided. Required format (example): 0.8-0.2")
        if frac_sum != 1.0:
            raise ValueError("Invalid split fractions were provided. They must sum to 1.")
    # ------------------------

    # updating the 'splits' argument
    # ------------------------
    parsed_arguments.splits = splits
    # ------------------------

    return parsed_arguments


# def classify_parse_args() -> argparse.Namespace:
#     """
#     Parse command line arguments for classifying phase.
    
#     Returns:
#         parsed_arguments (argparse.Namespace): populated Namespace: the arguments passed via 
#                                             command line are converted to objects and assigned 
#                                             as attributes of the namespace
#     """

#     parser = argparse.ArgumentParser(description='')

#     parser.add_argument('mode'
#                         , choices=['train', 'classify']
#                         , help='train the classifier or classify input image')

#     parser.add_argument('-a'
#                         , '--data_augmentation'
#                         , action='store_true'
#                         , help='data augmentation preprocessing is applied')

#     parser.add_argument('--dataset_folder'
#                         , type=str
#                         , default='./../data/'
#                         , help='(default=\'./../data/\') folder where to save the dataset or from where to load it (if mode == train)')
    
#     parser.add_argument('--model_path'
#                         , type=str
#                         , default='./../models/'
#                         , help='(default=\'./../models/\') path of the model to load from memory (if mode == classify)')

#     parser.add_argument('--model_name'
#                         , type=str
#                         , default=None
#                         , help='(default=None) name of the model to load from memory (if mode == classify)')

#     parser.add_argument('--splits'
#                         , type=str
#                         , default='0.8-0.2'
#                         , help='(default: 0.8-0.2) fraction of data to be used in training and validation set')

#     parser.add_argument('--batch_size'
#                         , type=int
#                         , default=64
#                         , help='(default: 64) mini-batch size')

#     parser.add_argument('--epochs'
#                         , type=int
#                         , default=10
#                         , help='(default: 10) number of training epochs')

#     parser.add_argument('--lr'
#                         , type=float
#                         , default=0.001
#                         , help='(default: 0.001) learning rate')

#     parser.add_argument('--workers'
#                         , type=int
#                         , default=3
#                         , help='(default: 3) number of working units used to load the data')

#     parser.add_argument('--device'
#                         , default='cpu'
#                         , type=str
#                         , help='(default: \'cpu\') device to be used for computations {cpu, cuda:0, cuda:1, ...}')

#     parsed_arguments = parser.parse_args()


#     # converting split fraction string to a list of floating point values ('0.8-0.2' => [0.8, 0.2])
#     # ------------------------
#     splits_string = str(parsed_arguments.splits)
#     fractions_string = splits_string.split('-')
#     if len(fractions_string) != 2:
#         raise ValueError("Invalid split fractions were provided. Required format (example): 0.8-0.2")
#     else:
#         splits = []
#         frac_sum = 0.
#         for fraction in fractions_string:
#             try:
#                 splits.append(float(fraction))
#                 frac_sum += splits[-1]
#             except ValueError:
#                 raise ValueError("Invalid split fractions were provided. Required format (example): 0.8-0.2")
#         if frac_sum != 1.0:
#             raise ValueError("Invalid split fractions were provided. They must sum to 1.")
#     # ------------------------

#     # updating the 'splits' argument
#     # ------------------------
#     parsed_arguments.splits = splits
#     # ------------------------

#     # checking presence of model folder and name if mode == classify
#     # ------------------------
#     if parsed_arguments.mode == 'classify' and (parsed_arguments.model_name is None or parsed_arguments.model_path is None):
#         raise ValueError("Model path and name must be provided if mode == 'classify'.")
#     # ------------------------

#     return parsed_arguments