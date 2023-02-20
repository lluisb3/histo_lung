from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv
from torch.utils.data import Dataset

thispath = Path(__file__).resolve()


class LungWSIDataset(Dataset):
    def __init__(self, dataset_set, dataset_mean=None, dataset_std=None, transform=None, seg_image=False):
        self.data_dir = thispath.parent.parent / "data" / "Mask_PyHIST"
        self.labels = pd.read_csv(self.data_dir.parent / "labels.csv")
        self.transform = transform
        self.mean = dataset_mean
        self.std = dataset_std
        self.dataset_set = dataset_set
        if self.dataset_set != 'test':
            self.metadata['Label']=self.metadata['Label'].astype(float)
        self.segmentation = seg_image

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.dataset_set == 'test':
            img_dir = str(self.data_dir / f"{self.metadata['Name'].iloc[idx]}.jpg")
        else:
            img_dir = str(self.data_dir / f"{self.metadata['Lesion Type'].iloc[idx]}/"
                         f"{self.metadata['Name'].iloc[idx]}.jpg")
        img = cv.imread(img_dir)  # read the image (BGR) using OpenCV (HxWxC)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # image now RGB
        if self.transform:
            img = z_score_norm(img,
                               mean=self.mean,
                               std=self.std,
                               only_non_zero=self.metadata['FOV presence'].iloc[idx])  # Image in float-32 (HxWxC)
            if self.segmentation:
                seg_im = cv.imread(str(self.data_dir)+f"_seg/{self.metadata['Lesion Type'].iloc[idx]}/"
                                                      f"{self.metadata['Name'].iloc[idx]}_seg.png", flags=0)
                img = np.concatenate((img, seg_im[:, :, np.newaxis]), axis=2)
            img = self.transform(img)

        sample = {'image': img, 'label': self.metadata['Label'].iloc[idx]}
        return sample