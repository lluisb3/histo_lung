import click 
from pathlib import Path
import pandas as pd

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
    Path(thispath.parent.parent / f'bat_files').mkdir(exist_ok=True, parents=True)

    datadir = Path("/mnt/nas4/datasets/ToReadme/ExaMode_Dataset1/AOEC")

    svs_files = [i for i in datadir.rglob("*.svs") if "LungAOEC" in str(i)]

    labels = pd.read_csv(Path(thispath.parent / "data" / "lung_data" / "he_images.csv"))
    names = labels["file_name"].values

    he_svs_files = []
    for name in names:
        for file in svs_files:
            if file.stem in name:
                he_svs_files.append(file)

    with open(
            Path(thispath.parent.parent / Path(
                f"bat_files/bash_{name_experiment}.sh")), 'w') as f:
        f.write(
            f"#!/bin/bash \n\n" 
        )
        for file in svs_files:
            bash_line = f'python pyhist.py --content-threshold 0.05 --patch-size 64 --output-downsample 16 --info "verbose" --save-mask {file}\n'

            f.write(bash_line)
        f.write(f"ECHO End Preprocessing for: {name_experiment} \n")
        f.write("PAUSE")


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