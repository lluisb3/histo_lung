from pathlib import Path
import cv2 as cv
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from utils import check_corners

thispath = Path(__file__).resolve()


def binary_mask_pyhist():

    maskdir = Path(thispath.parent.parent / "data" / "Mask_PyHIST_v2")

    colorful_masks = [i for i in maskdir.rglob("*.ppm")]

    for mask in tqdm(colorful_masks, desc=f"Saving binary masks in Mask_PyHIST"):

        color_mask = cv.imread(str(mask))

        # Mask from PyHIST to binary mask
        copy = color_mask.copy()

        most_frequent = check_corners(copy)
        copy[(copy != most_frequent).any(axis=-1)] = 1
        copy[(copy == most_frequent).all(axis=-1)] = 0
        binary_mask = copy[:, :, 0]

        cv.imwrite(f"{maskdir}/{mask.parent.parent.stem}/{mask.parent.stem}/binary_{mask.parent.stem}.png",
                   binary_mask, [cv.IMWRITE_PNG_BILEVEL, 1])

    print(f"Binary masks created from PyHIST and saved in {maskdir}")


def main():
    binary_mask_pyhist()


if __name__ == "__main__":
    main()
