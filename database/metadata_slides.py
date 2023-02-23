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

    subdirs = natsorted([e for e in maskdir.iterdir() if e.is_dir()])
    list_dirs = []
    for dir in subdirs:
        list_dirs += [i for i in dir.iterdir() if i.is_dir()]
    names = []
    for path in list_dirs:
        names.append(path.stem)

    he_svs_files = []
    for name in names:
        for file in svs_files:
            if file.stem in name:
                he_svs_files.append(file)

    print(f"Number of svs files for the metadata: {len(he_svs_files)}")

    header = ["level_dimensions", "level_downsamples", "magnifications", "mpp",
              "number_patches_pyhist", "patch_shape", "center", "mean", "std", "mask_shape_stadistics"]

    metadata = pd.DataFrame(columns=header)

    for svs_file in tqdm(he_svs_files, desc="Metadata .csv file in progress"):

        patchdir = Path(maskdir / svs_file.parent.stem / svs_file.stem / f"{svs_file.stem}_tiles")
        binary_path = Path(maskdir / svs_file.parent.stem / svs_file.stem / f"binary_{svs_file.stem}.png")

        number_patches = len(os.listdir(patchdir))
            
        patch = cv.imread(str(Path(patchdir / os.listdir(patchdir)[0] )))
        patch_shape = patch.shape
        
        center = svs_file.parent.stem.split("_")[0]

        slide = openslide.OpenSlide(str(svs_file))

        level_dimensions = slide.level_dimensions
        mpp = slide.properties['openslide.mpp-x']
        level_downsamples = slide.level_downsamples
        mags = available_magnifications(mpp, level_downsamples)

        binary_mask = cv.imread(str(binary_path))
        binary_mask[binary_mask == 255] = 1
        mask_shape = binary_mask.shape
        binary_mask = cv.resize(binary_mask, (int(mask_shape[1]*0.5), int(mask_shape[0]*0.5)))
        mask_shape = binary_mask.shape

        thumbnail = slide.get_thumbnail((mask_shape[1], mask_shape[0]))

        thumbnail_data = np.array(thumbnail)
        if thumbnail_data.shape != mask_shape:
            thumbnail_data = cv.resize(thumbnail_data, (mask_shape[1], mask_shape[0]))

        thumb_data_masked = np.ma.array(thumbnail_data, mask=np.logical_not(binary_mask))
        mean_thumb_data = np.mean(thumb_data_masked, axis=(0, 1))
        std_thumb_data = np.std(thumb_data_masked, axis=(0, 1))
        
        metadata.loc[svs_file.stem] = [level_dimensions, level_downsamples, mags,
                                       mpp, number_patches, patch_shape, center,
                                       mean_thumb_data, std_thumb_data, mask_shape]

    metadata.to_csv(f"{maskdir.parent}/metadata_slides.csv")
    print(f"metadata_slides.csv created in {maskdir.parent}")


def main():
    metadata_csv()


if __name__ == "__main__":
    main()
