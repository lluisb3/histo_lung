from pathlib import Path
import os
import pandas as pd
import numpy as np
import openslide
from tqdm import tqdm
import cv2 as cv
import click
from utils import available_magnifications

thispath = Path(__file__).resolve()


def metadata_one(image_index):


    datadir = Path("/mnt/nas4/datasets/ToReadme/ExaMode_Dataset1/AOEC")
    maskdir = Path(thispath.parent.parent / "data" / "Mask_PyHIST")

    svs_file = [i for i in datadir.rglob("*.svs") if "LungAOEC" in str(i) and image_index in str(i)][0]


    header = ["level_dimensions", "level_downsamples", "magnifications", "mpp",
                "number_patches_pyhist", "patch_shape", "center", "mean", "std", "mask_shape_stadistics"]

    metadata = pd.DataFrame(columns=header)


    print(f"== {svs_file.stem} ==")
    patchdir = Path(maskdir / svs_file.parent.stem / svs_file.stem / f"{svs_file.stem}_tiles")
    resultdir = Path(maskdir / svs_file.parent.stem / svs_file.stem)

    number_patches = len(os.listdir(patchdir))

    patches_metadata = pd.read_csv(Path(resultdir / "tile_selection.tsv"), sep='\t').set_index("Tile")
    patch_shape = patches_metadata.iloc[0]["Width"]

    center = svs_file.parent.stem.split("_")[0]

    slide = openslide.OpenSlide(str(svs_file))

    level_dimensions = slide.level_dimensions
    mpp = slide.properties['openslide.mpp-x']
    level_downsamples = slide.level_downsamples
    mags = available_magnifications(mpp, level_downsamples)

    binary_mask = cv.imread(str(resultdir / f"binary_{svs_file.stem}.png"))
    binary_mask[binary_mask == 255] = 1

    thumbnail = slide.get_thumbnail(level_dimensions[0])

    thumbnail_data = np.array(thumbnail)
    thumbnail_shape = thumbnail_data.shape

    binary_mask = cv.resize(binary_mask, (thumbnail_shape[1], thumbnail_shape[0]))

    thumb_data_masked = np.ma.array(thumbnail_data, mask=np.logical_not(binary_mask))
    mean_thumb_data = np.mean(thumb_data_masked, axis=(0, 1))
    std_thumb_data = np.std(thumb_data_masked, axis=(0, 1))

    metadata.loc[svs_file.stem] = [level_dimensions, level_downsamples, mags,
                                    mpp, number_patches, patch_shape, center,
                                    mean_thumb_data, std_thumb_data, thumbnail_shape]

    outputdir = Path(maskdir.parent / f"metadata_for_one_big_{image_index}.csv")
    metadata.to_csv(outputdir)


@click.command()
@click.option(
    "--image_index",
    default="0000",
    prompt="Image index",
    help="Image index",
)
def main(image_index):
	metadata_one(image_index)
	
if __name__ == "__main__":
    main()
