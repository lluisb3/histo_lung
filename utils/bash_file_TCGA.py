import click 
from pathlib import Path
import openslide
from utils import available_magnifications
from tqdm import tqdm

thispath = Path(__file__).resolve()

print()

def bash_file(name_experiment):
    """
    Function to create a .txt file ready to be run as elastix file in the console to perform the registration of
    a set of landmarks using the transformation used to perform the registration using transformix command.
    Parameters
    ----------
    name_experiment: Name of the bash file

    Returns
    -------
    A .sh bash file to perform the preprocessing with PyHIST
    """
    Path(thispath.parent.parent / f'bash_files').mkdir(exist_ok=True, parents=True)

    datadir = Path("/mnt/nas4/datasets/ToReadme/ExaMode_Dataset1/TCGA/Lung")

    outputdir = Path("/mnt/nas6/data/lung_tcga/Mask_PyHIST" )

    svs_files = [i for i in datadir.rglob("*.svs")]

    with open(
            Path(thispath.parent.parent / Path(
                f"bash_files/bash_{name_experiment}.sh")), 'w') as f:
        f.write(
            f"#!/bin/bash \n\n" 
        )
        
        for file in tqdm(svs_files):
            slide = openslide.OpenSlide(str(file))
            magnification = slide.properties["aperio.AppMag"]
            if magnification == "40":
                downsample = 4
            elif magnification == "20":
                downsample = 2
            
            bash_line = f"python3 pyhist.py --method graph --mask-downsample 16 --output-downsample {downsample} " \
                        f"--tilecross-downsample 32 --corners 1111 --borders 0000 --percentage-bc 1 " \
                        f"--k-const 1000 --minimum_segmentsize 1000 --info 'verbose' --content-threshold 0.2 " \
                        f"--patch-size 256 --save-patches --save-mask --save-tilecrossed-image " \
                        f"--output {outputdir}/{file.parent.stem}/ {file}\n"
            f.write(bash_line)




@click.command()
@click.option(
    "--name",
    default="Example",
    prompt="Name of the bash file",
    help=
    "Choose name for the bash_file",
)
def main(name):
        bash_file(name)


if __name__ == "__main__":
    main()
