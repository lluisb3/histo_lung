from pathlib import Path
import pandas as pd
import numpy as np
import click
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

thispath = Path(__file__).resolve()


def data_splits(k, test_size):
    """Performs k-fold-crossvalidation on the LungAOEC dataset with a desired
    number of folds (k). 

    Parameters
    ----------
    k : int
        Number of folds
    test_size : float
        Percentage of images that belongs to the test set, rest of images
        will go to the train set (from 0 to 1) 
    """

    datadir = Path(thispath.parent.parent / "data")

    csv_dataset_AOEC = Path(datadir / "labels.csv")

    # read data
    dataset_AOEC = pd.read_csv(csv_dataset_AOEC, sep=',', header=0).values

    mskf = MultilabelStratifiedShuffleSplit(n_splits=k,
                                            test_size=test_size,
                                            random_state=33)

    images = dataset_AOEC[:, 0]

    labels = dataset_AOEC[:, 1:]

    header = ["images_train", "images_test", "labels_train", "labels_test"]
    folds = pd.DataFrame(columns=header)
    i = 0

    for train_index, test_index in mskf.split(images, labels):
        images_train, images_test = images[train_index], images[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        folds.loc[i] = [images_train.tolist(), images_test.tolist(),
                        labels_train.tolist(), labels_test.tolist()]
        i += 1

    print(f"Datasplit labels TRAIN: {np.sum(labels_train, axis=0)}"
          f"Datasplit labels TEST: {np.sum(labels_test, axis=0)}")

    folds.to_csv(Path(datadir / f"{k}_fold_crossvalidation_data_split.csv"))

    print(f"{k}_fold_crossvalidation_data_split.csv in {datadir}")


@click.command()
@click.option(
    "--k",
    default=10,
    prompt="Number of splits (k)",
    help="Specify number of splits to perform k-fold-crossvalidation",
)
@click.option(
    "--test_size",
    default=0.2,
    prompt="Percentage of the test dataset 0 to 1",
    help="Percentage of the test dataset, must be from 0 to 1",
)
def main(k, test_size):
    data_splits(k, test_size)


if __name__ == "__main__":
    main()
