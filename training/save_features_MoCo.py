from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from natsort import natsorted
from training import Encoder, ModelOption, yaml_load
from database import Dataset_instance
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from utils import timer
import click

thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Load PyTorch model

def MoCo_features(exp_name):
    experiment_name = exp_name

    modeldir = Path(thispath.parent.parent / "trained_models" / "MoCo" / experiment_name)

    cfg = yaml_load(modeldir / f"config_{experiment_name}.yml")

    model = ModelOption(cfg.model.model_name,
                    cfg.model.num_classes,
                    freeze=cfg.model.freeze_weights,
                    num_freezed_layers=cfg.model.num_frozen_layers,
                    dropout=cfg.model.dropout,
                    embedding_bool=cfg.model.embedding_bool
                    )    

    # Encoder and momentum encoder
    moco_dim = cfg.training.moco_dim

    encoder = Encoder(model, dim=moco_dim).to(device)

    # checkpoint = torch.load(modeldir / cfg.dataset.magnification / cfg.model.model_name / f"{experiment_name}.pt")
    checkpoint = torch.load(modeldir / cfg.dataset.magnification / cfg.model.model_name / "MoCo.pt")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    loss = checkpoint["loss"]
    epoch = checkpoint["epoch"] + 1

    print(f"Loaded encoder using as backbone {cfg.model.model_name} with a best loss of {loss} at Epoch {epoch}")

    # Load patches
    pyhistdir = Path(datadir / "Mask_PyHIST_v2")

    dataset_path = natsorted([i for i in pyhistdir.rglob("*_densely_filtered_paths.csv") if "LungAOEC" in str(i)])

    number_patches = 0
    path_patches = []
    patches_names = []
    for wsi_patches in tqdm(dataset_path, desc="Selecting patches to check model"):

        csv_instances = pd.read_csv(wsi_patches).to_numpy()

        path_patches.extend(csv_instances)
        for instance in csv_instances:
            patches_names.append(str(instance).split("/")[-1])

    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev),
            transforms.Resize(size=(model.resize_param, model.resize_param),
            antialias=True)
        ])

    params_instance = {'batch_size': 1035,
                       'shuffle': False,
                       'pin_memory': True,
                       'num_workers': 5}

    instances = Dataset_instance(path_patches, transform=None, preprocess=preprocess)
    generator = DataLoader(instances, **params_instance)

    encoder.eval()
    # feature_patches_dict = {}
    start = time.time()
    feature_matrix = np.zeros([len(path_patches), moco_dim], dtype=float)
    with torch.no_grad():

        for i, (x_q, _) in tqdm(enumerate(generator)):
            x_q = x_q.to(device, non_blocking=True)

            q = encoder(x_q)
            q = q.squeeze().cpu().numpy()

            feature_matrix[i*1035:(i+1)*1035, :] = q

    df_feature = pd.DataFrame(feature_matrix, index=patches_names, columns=range(moco_dim))
    df_feature.to_csv(Path(modeldir / f"features_{experiment_name}.csv"))
    message = timer(start=start, end=time.time())
    print(message)


@click.command()
@click.option(
    "--exp_name",
    default="MoCo_try_Adam",
    prompt="Name of the MoCo model to extrcat features",
    help="Name of the MoCo model to extrcat features",
)
def main(exp_name):
    MoCo_features(exp_name)


if __name__ == '__main__':
    main()
