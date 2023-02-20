from pathlib import Path
import os
import pandas as pd
import numpy as np
import openslide
from tqdm import tqdm
import cv2 as cv
from natsort import natsorted
from utils import available_magnifications

thispath = Path(__file__).resolve()

def metadata_csv():
    datadir = Path("/mnt/nas4/datasets/ToReadme/ExaMode_Dataset1/AOEC")
    maskdir = Path(thispath.parent.parent / "data" / "Mask_PyHIST")

    svs_files = natsorted([i for i in datadir.rglob("*.svs") if "LungAOEC" in str(i)], key=str)

    labels = pd.read_csv(Path(thispath.parent.parent / "data" / "lung_data" / "he_images.csv"))
    names = natsorted(labels["file_name"].values, key=str)

    he_svs_files = []
    for name in names:
        for file in svs_files:
            if file.stem in name:
                he_svs_files.append(file)

    # for mask, file in natsorted(zip(binary_masks, he_svs_files), key=str):
    #      print(file)
    #      print(mask)

    print(f"Number of svs files for the metadata: {len(he_svs_files)}")

    header = ["level_dimensions", "level_downsamples", "magnifications", "mpp", "number_patches_PyHist"]

    metadata = pd.DataFrame(columns=header)

    for svs_file in tqdm(he_svs_files, desc="Metadata .csv file in progress"):
        
        number_patches = len(os.listdir(Path(maskdir / svs_file.parent.stem / svs_file.stem / f"{svs_file.stem}_tiles")))

        slide = openslide.OpenSlide(str(svs_file))

        level_dimensions = slide.level_dimensions
        mpp = slide.properties['openslide.mpp-x']
        level_downsamples = slide.level_downsamples
        mags = available_magnifications(mpp, level_downsamples)
        metadata.loc[svs_file.stem] = [level_dimensions, level_downsamples, mags, mpp, number_patches]

    metadata.to_csv(f"{maskdir.parent}/metadata_slides.csv")
    print(f"metadata_slides.csv created in {maskdir.parent}")


def main():
    metadata_csv()


if __name__ == "__main__":
    main()
