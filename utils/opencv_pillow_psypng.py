from pathlib import Path
import pandas as pd
import cv2 as cv
from PIL import Image
from tqdm import tqdm
import numpy as np
import pyspng
import albumentations as A
from natsort import natsorted
from time import time
from utils import timer

thispath = Path(__file__).resolve()


def compare_read_images():
    pyhistdir = Path(thispath.parent.parent / "data" / "Mask_PyHIST_v2")

    dataset_path = natsorted([i for i in pyhistdir.rglob("*_densely_filtered_paths.csv")], )

    number_patches = 0
    path_patches = []
    for wsi in tqdm(dataset_path, desc="Selecting patches"):

        csv_instances = pd.read_csv(wsi).to_numpy()
        # csv_instances.values
        l_csv = len(csv_instances)
        
        number_patches = number_patches + l_csv
        path_patches.extend(csv_instances)

    start = time()
    for image in tqdm(path_patches, desc="Read with openCV"):
        img = cv.imread(image[0])
    print(f"Time to read {len(path_patches)} with openCV")
    timer(start, time())

    start = time()
    for image in tqdm(path_patches, desc="Read with Pillow"):
        img = Image.open(image[0])
        pix = np.array(img)
    print(f"Time to read {len(path_patches)} with Pillow")
    timer(start, time())

    start = time()
    for image in tqdm(path_patches, desc="Read with pyspng"):
        with open(image[0], 'rb') as fin:
            img = pyspng.load(fin.read())
    print(f"Time to read {len(path_patches)} with pyspng")
    timer(start, time())


def main():
    compare_read_images()


if __name__ == "__main__":
    main()

