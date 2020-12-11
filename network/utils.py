import requests
import os
import gzip
import shutil
import torch.tensor
import struct

def is_downloadable(url: str) -> bool:
    """
    Does the url contain a downloadable resource for our project.

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


def download(url: str, folder: str, filename: str) -> None:
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

        print("Checking presence of {} ...".format(compressed_file_path))
        if not os.path.exists(compressed_file_path): # if no downloads present

            print("Downloading {} ...".format(compressed_file_path))
            
            # write the files
            with open(compressed_file_path, 'wb') as f:
                f.write(r.raw.data)
        
        else:   # if the files have already been downloaded
            print("Already downloaded.")


        print("Checking presence of uncompressed file {} ...".format(file_path))
        if not os.path.exists(file_path):   # if uncompressed file is present 

            print("Extracting {} ...".format(file_path))
            
            # open the compressed file
            with gzip.open(compressed_file_path, 'rb') as f_in:
                # open the uncompressed file to be filled
                with open(file_path, 'wb') as f_out:
                    # fill the uncompressed file
                    shutil.copyfileobj(f_in, f_out)
        
        else:   # if the files have already been downloaded
            print("Already extracted.")


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