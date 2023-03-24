import torch
from torch.utils.data import Dataset
from pathlib import Path
import pyspng
from PIL import Image
import numpy as np
  
thispath = Path(__file__).resolve()


class Dataset_instance(Dataset):

    def __init__(self, wsi_path_patches, transform=None, preprocess=None):

        self.wsi_path_patches = wsi_path_patches
        self.transform = transform
        self.preprocess = preprocess


    def __len__(self):
        return len(self.wsi_path_patches)

    def __getitem__(self, index):
        
        # Load the patch image saved as png (key)
        # with open(self.wsi_path_patches[index][0], 'rb') as fin:
        #     key =  pyspng.load(fin.read())
        # open method used to open different extension image file
        key = np.array(Image.open(self.wsi_path_patches[index][0]))  

        if self.transform:
            query = self.transform(image=key)['image']
        else:
            query = key

        if self.preprocess:
            query = self.preprocess(query).type(torch.FloatTensor)
            key = self.preprocess(key).type(torch.FloatTensor)
        
        return key, query


class Dataset_bag(Dataset):
    def __init__(self, list_IDs, labels):

        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        return self.list_IDs[index]


class Dataset_instance_MIL(Dataset):

    def __init__(self, wsi_path_patches, transform=None, preprocess=None):
        self.wsi_path_patches = wsi_path_patches
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index][0]
        # Load data and get label
        with open(ID, 'rb') as fin:
            input_tensor = pyspng.load(fin.read())
        #img.close()

        if self.transform:
            input_tensor = self.transform(image=input_tensor)['image']
            #X = pipeline_transform_local(image=X)['image']

        #data transformation
        if self.preprocess:
            input_tensor = self.preprocess(input_tensor).type(torch.FloatTensor)
        
        return input_tensor


class Dataset_bag_MIL(Dataset):

    def __init__(self, list_IDs, labels):
        self.list_IDs = list_IDs
        self.labels = labels
		
    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        return self.list_IDs[index], self.labels[index]
