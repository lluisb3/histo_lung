import torch
from torch.utils.data import Dataset
from pathlib import Path
import pyspng
from PIL import Image
import numpy as np
import cv2 as cv
  
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
        # key = np.array(Image.open(self.wsi_path_patches[index][0]))
        key = cv.imread(self.wsi_path_patches[index][0])
        key = cv.cvtColor(key, cv.COLOR_BGR2RGB)

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
        return len(self.wsi_path_patches)

    def __getitem__(self, index):
        # Select sample
        wsi_id = self.wsi_path_patches[index][0]
        # Load data and get label
        with open(wsi_id, 'rb') as fin:
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


class Balanced_Multimodal(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None, alpha = 0.5):

        self.indices = list(range(len(dataset)))             if indices is None else indices

        self.num_samples = len(self.indices)             if num_samples is None else num_samples

        class_sample_count = [0,0,0,0]
        
        # labels = np.array(dataset[:,1])
        class_sample_count = np.sum(dataset[:, 1:], axis=0)

        min_class = np.argmin(class_sample_count)
        class_sample_count = np.array(class_sample_count)
        weights = []
        for c in class_sample_count:
            weights.append((c/class_sample_count[min_class]))

        ratio = np.array(weights).astype(np.float)

        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            for l in label:
                if l in label_to_count:
                    label_to_count[l] += 1
                else:
                    label_to_count[l] = 1

        weights = []

        for idx in self.indices:
            c = 0
            for j, l in enumerate(self._get_label(dataset, idx)):
                c = c+(1/label_to_count[l])#*ratio[l]

            weights.append(c/(j+1))
            #weights.append(c)
			
        self.weights_original = torch.DoubleTensor(weights)

        self.weights_uniform = np.repeat(1/self.num_samples, self.num_samples)

        #print(self.weights_a, self.weights_b)

        beta = 1 - alpha
        self.weights = (alpha * self.weights_original) + (beta * self.weights_uniform)


    def _get_label(self, dataset, idx):
        labels = np.where(dataset[idx, 1:]==1)[0]
        #print(labels)
        #labels = dataset[idx,2]
        return labels

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
