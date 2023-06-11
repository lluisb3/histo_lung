from pathlib import Path
import numpy as np
import cv2 as cv
import pandas as pd
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import openslide
import time
from preprocessing import eval_histogram_threshold, get_histogram

thispath = Path(__file__).resolve()

datadir = Path("/mnt/nas6/data/ExaMode_data/Radboudumc")


def filter_patches(list_dirs, maskdir):

    for filename in tqdm(list_dirs, desc="Filtering patches from PyHIST"):
        binary_mask = cv.imread(str(Path(maskdir / filename.stem /
                                         f"binary_{filename.stem}.png")))
        binary_mask[binary_mask == 255] = 1
        mask_shape = binary_mask.shape
        binary_mask = cv.resize(binary_mask, (int(mask_shape[1]*0.125), int(mask_shape[0]*0.125)))
        mask_shape = binary_mask.shape

        slide = openslide.OpenSlide(str(Path(datadir.parent / "lung" /f"{filename.stem}.tif" )))
        thumbnail = slide.get_thumbnail((mask_shape[1], mask_shape[0]))

        thumbnail_data = np.array(thumbnail)
        if thumbnail_data.shape != mask_shape:
            thumbnail_data = cv.resize(thumbnail_data, (mask_shape[1], mask_shape[0]))

        lower, upper = eval_histogram_threshold(binary_mask, thumbnail_data)
        print(f"Set an lower threshold of {lower} and upper {upper} to compute the histogram")

        patches_metadata = pd.read_csv(Path(filename / "tile_selection.tsv"), sep='\t').set_index("Tile")
        
        patches_path = [i for i in filename.rglob("*.png") if "tiles" in str(i)]

        if len(patches_path) > 0:

            patch = cv.imread(str(patches_path[0]))
            patch_shape = patch.shape
            total_pixels_patch = patch_shape[0] * patch_shape[1]
            filtered_patches = []
            names = []
            all_row = []
            all_col = []

            for image_patch in tqdm(patches_path, desc=f"Filtering patches of {filename.stem}"):

                image = cv.imread(str(image_patch))
                gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                histo = get_histogram(gray_image, lower, upper)
                
                total_pixels_in_range = np.sum(histo)
                
                if (total_pixels_in_range > 0.5 * total_pixels_patch):
                    name = image_patch.stem
                    names.append(name)
                    all_row.append(patches_metadata.loc[name]['Row'])
                    all_col.append(patches_metadata.loc[name]['Column'])
                    filtered_patches.append(image_patch)
                
            # Create .csv with metadata information of the filtered patches
            outputdir_metadata = Path(filename / f"{filename.stem}_densely_filtered_metadata_v2.csv")
            
            file_metadata = {'patch_name':names,'row':all_row,'column':all_col}
            df_metadata = pd.DataFrame.from_dict(file_metadata)
            df_metadata.sort_values(by='patch_name', axis=0, ascending=True, inplace=True)
            df_metadata.to_csv(outputdir_metadata, index=False)

            # Create .csv with filtered parches path
            outputdir_paths = Path(filename / f"{filename.stem}_densely_filtered_paths_v2.csv")
            file_path = {'filtered_patch_path': filtered_patches}
            df_paths = pd.DataFrame.from_dict(file_path)
            df_paths.sort_values(by='filtered_patch_path', axis=0, ascending=True, inplace=True)
            df_paths.to_csv(outputdir_paths, index=False)

            print(f"Filtered patches: {len(filtered_patches)} from a total of {len(patches_path)}")
            print(f"Filtered .csv for {filename.stem} saved on {outputdir_paths.parent}")
        else:
            filtered_patches = []
            names = []
            all_row = []
            all_col = []
            outputdir_metadata = Path(filename / f"{filename.stem}_densely_filtered_metadata_v2.csv")
            
            file_metadata = {'patch_name':names,'row':all_row,'column':all_col}
            df_metadata = pd.DataFrame.from_dict(file_metadata)
            df_metadata.sort_values(by='patch_name', axis=0, ascending=True, inplace=True)
            df_metadata.to_csv(outputdir_metadata, index=False)

            # Create .csv with filtered parches path
            outputdir_paths = Path(filename / f"{filename.stem}_densely_filtered_paths_v2.csv")
            file_path = {'filtered_patch_path': filtered_patches}
            df_paths = pd.DataFrame.from_dict(file_path)
            df_paths.sort_values(by='filtered_patch_path', axis=0, ascending=True, inplace=True)
            df_paths.to_csv(outputdir_paths, index=False)

            print(f"Filtered patches: {len(filtered_patches)} from a total of {len(patches_path)}")
            print(f"Filtered .csv for {filename.stem} saved on {outputdir_paths.parent}")


def main():

    start_time = time.time()

    maskdir = Path(thispath.parent.parent / "data" / "Mask_PyHIST")

    subdirs = natsorted([e for e in maskdir.iterdir() if e.is_dir()])

    subdirs = subdirs[:137]

    filter_patches(subdirs, maskdir)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time}")


if __name__ == "__main__":
	main()
