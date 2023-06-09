from pathlib import Path
import pandas as pd
import numpy as np
import click
from utils import create_folds

thispath = Path(__file__).resolve()


def data_splits(k):
    """Performs k-fold-crossvalidation on the LungAOEC dataset with a desired
    number of folds (k).

    Parameters
    ----------
    k : int
        Number of folds
    """

    datadir = Path(thispath.parent.parent / "data")

    csv_ids = Path(datadir / "patients_ID.csv")
    csv_dataset_AOEC = Path(datadir / "manual_labels.csv")

    # read data
    dataset_AOEC = pd.read_csv(csv_dataset_AOEC,
                                sep=',', 
                                index_col=0, 
                                dtype={"image_num":str})

    patients_id = pd.read_csv(csv_ids,
                                sep=',', 
                                index_col=0, 
                                dtype={"FILENAME": str, "ID": str})

    df = patients_id.drop_duplicates(subset='ID', keep="first")
    patients = df.values

    folds = create_folds(patients, k)
    header = ["images_train", "images_validation", "labels_train", "labels_validation"]
    folds_dataset = pd.DataFrame(columns=header)

    for i in range(k):
        train_patients = folds[:i] + folds[i+1:]
        train_patients = [item for sublist in train_patients for item in sublist]
        train_patients = [item for sublist in train_patients for item in sublist]
        validation_patinets = folds[i]
        validation_patinets = [item for sublist in validation_patinets for item in sublist]

        train_filenames = patients_id[patients_id['ID'].isin(train_patients)].index
        validation_filenames = patients_id[patients_id['ID'].isin(validation_patinets)].index
        train = dataset_AOEC[dataset_AOEC.index.isin(train_filenames)]
        validation = dataset_AOEC[dataset_AOEC.index.isin(validation_filenames)]

        images_train = train.index.to_list()
        labels_train = train.values.tolist()
        images_validation = validation.index.to_list()
        labels_validation = validation.values.tolist()

        folds_dataset.loc[i] = [images_train, images_validation, labels_train, labels_validation]

        print(f"Number WSI TRAIN: {len(images_train)}, Number WSI VALID: {len(images_validation)}")
        print(f"Datasplit labels TRAIN: {np.sum(labels_train, axis=0)}, "
            f"Datasplit labels TEST: {np.sum(labels_validation, axis=0)}")

    folds_dataset.index.name = "fold"
    folds_dataset.to_csv(Path(datadir / f"{k}_fold_crossvalidation_data_split.csv"))

    print(f"{k}_fold_crossvalidation_data_split.csv in {datadir}")


@click.command()
@click.option(
    "--k",
    default=5,
    prompt="Number of splits (k)",
    help="Specify number of splits to perform k-fold-crossvalidation",
)
def main(k):
    data_splits(k)


if __name__ == "__main__":
    main()
