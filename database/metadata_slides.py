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


def metadata_slides_csv():
    datadir = Path("/mnt/nas6/data/lung_tcga/data")
    maskdir = Path(datadir.parent / "Mask_PyHIST_tif")

    svs_files = natsorted([i for i in datadir.rglob("*.tif")], key=str)

    subdirs = natsorted([e for e in maskdir.iterdir() if e.is_dir()])
    list_dirs = []
    for dir in subdirs:
        list_dirs += [i for i in dir.iterdir() if i.is_dir()]

    print(f"Number of svs files for the metadata: {len(svs_files)}")

    header = ["level_dimensions", "level_downsamples", "number_patches",
              "number_filtered_patches", "patch_shape", "center"]

    metadata = pd.DataFrame(columns=header)
    metadata.index.name = "ID wsi"

    for svs_file in tqdm(svs_files, desc="Metadata .csv file in progress"):
        patchdir = Path(maskdir / svs_file.parent.stem / svs_file.stem / f"{svs_file.stem}_tiles")
        resultdir = Path(maskdir / svs_file.parent.stem / svs_file.stem)

        number_patches = len(os.listdir(patchdir))

        patches_metadata = pd.read_csv(Path(resultdir / "tile_selection.tsv"), sep='\t').set_index("Tile")
        patch_shape = patches_metadata.iloc[0]["Width"]

        center = svs_file.parent.parent.stem

        slide = openslide.OpenSlide(str(svs_file))

        level_dimensions = slide.level_dimensions
        level_downsamples = slide.level_downsamples

        df_csv = pd.read_csv(Path(resultdir / f"{svs_file.stem}_densely_filtered_metadata.csv"))
        filtered_patches = df_csv["patch_name"].count()

        metadata.loc[svs_file.stem] = [level_dimensions, level_downsamples, number_patches, filtered_patches,
                                       patch_shape, center]

    metadata.sort_index(axis=0, ascending=True, inplace=True)
    metadata.to_csv(Path(maskdir / "metadata_slides_v2.csv"))
    print(f"metadata_slides.csv created in {maskdir}")


def main():
    metadata_slides_csv()


if __name__ == "__main__":
    main()
