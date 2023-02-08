import click 
from pathlib import Path

thispath = Path(__file__).resolve()

print()

def bash_file(name_experiment):
    """
    Function to create a .txt file ready to be run as elastix file in the console to perform the registration of
    a set of landmarks using the transformation used to perform the registration using transformix command.
    Parameters
    ----------
    name_experiment: Name of the experiment to save transformix results and get results coming from elastix
    registration.
    parameter: Name of the folder with the parameters in elastix/parameters
    dataset_option: set to train if you use any dataset of train images (train, train_Normalized_CLAHE, etc), set to
    test if you use any test dataset.

    Returns
    -------
    A .txt file in elastix/bat_files with name transformix_name_experiment to run in the console the transformix
    transformation of a set of landmarks. The new landmarks results are saved in the path
    elastix/Outputs_experiments_transformix/name_experiment
    """
    Path(thispath.parent.parent / f'bat_files').mkdir(exist_ok=True, parents=True)

    datadir = Path("/mnt/nas4/datasets/ToReadme/ExaMode_Dataset1/AOEC")

    svs_files = [i for i in datadir.rglob("*.svs") if "LungAOEC" in str(i)]

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
    "--name_experiment",
    default="Example",
    help=
    "Chose to create an elastix or transformix system file. If elastix the following parameters are needed:"
    "name_experiment, parameter, dataset_option and optionally mask and mask_name"
    "If transformix the following parameters are meeded:"
    "name_experiment, parameters, dataset_option",
)
def main(name_experiment):
        bash_file(name_experiment)


if __name__ == "__main__":
    main()