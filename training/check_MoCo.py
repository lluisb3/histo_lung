from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from natsort import natsorted
from training import Encoder, ModelOption, yaml_load, cosine_similarity
from database import Dataset_instance
from torch.utils.data import DataLoader
from torchvision import transforms
from itertools import combinations
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import seaborn as sns

thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load PyTorch model
experiment_name = "MoCo_resnet34_scheduler_51015"

modeldir = Path(thispath.parent.parent / "trained_models" / "MoCo" / experiment_name)

cfg = yaml_load(modeldir / f"config_{experiment_name}.yml")

model = ModelOption(cfg.model.model_name,
                cfg.model.num_classes,
                freeze=cfg.model.freeze_weights,
                num_freezed_layers=cfg.model.num_freezed_layers,
                dropout=cfg.model.dropout,
                embedding_bool=cfg.model.embedding_bool
                )    

# Encoder and momentum encoder
moco_dim = cfg.training.moco_dim

encoder = Encoder(model, dim=moco_dim).to(device)

# checkpoint = torch.load(modeldir / cfg.dataset.magnification / cfg.model.model_name / f"{experiment_name}.pt")
checkpoint = torch.load(modeldir / cfg.dataset.magnification / cfg.model.model_name / f"{experiment_name}.pt")
encoder.load_state_dict(checkpoint["encoder_state_dict"])
loss = checkpoint["loss"]
epoch = checkpoint["epoch"] + 1

print(f"Loaded encoder using as backbone {cfg.model.model_name} with a best loss of {loss} at Epoch {epoch}")

# Load patches
pyhistdir = Path(datadir / "Mask_PyHIST_v2")

selected_wsi = ["000030303300314205", 
                "000030494900323685", 
                "000030689200333236", 
                "000030689500332896",
                "000030734200335036",
                "000030734200335038"]

dataset_path = natsorted([i for i in pyhistdir.rglob("*_densely_filtered_paths.csv") 
                          if selected_wsi[0] in str(i) 
                          or selected_wsi[1] in str(i)
                          or selected_wsi[2] in str(i)
                          or selected_wsi[3] in str(i)
                          or selected_wsi[4] in str(i)
                          or selected_wsi[5] in str(i)])

path_patches = []
for wsi_patches in tqdm(dataset_path, desc="Selecting patches to check model"):

    csv_instances = pd.read_csv(wsi_patches).to_numpy()
    
    path_patches.extend(csv_instances)

# Select patches to compute similarity
csvdir = Path(datadir.parent / "csv_patch_similarity")

csv_path = natsorted([i for i in csvdir.rglob("*.csv") if "Patch" in str(i) 
                      if selected_wsi[0] in str(i) 
                      or selected_wsi[1] in str(i)
                      or selected_wsi[2] in str(i)
                      or selected_wsi[3] in str(i)
                      or selected_wsi[4] in str(i)
                      or selected_wsi[5] in str(i)])
inspection_dict = {}
for csv_file in csv_path:
    inspection_dict[csv_file.stem.split("-")[1]] = {}
    csv_data = pd.read_csv(csv_file)
    for column in csv_data:
        inspection_dict[csv_file.stem.split("-")[1]][column] = csv_data[column].dropna().values

preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev),
        transforms.Resize(size=(model.resize_param, model.resize_param),
        antialias=True)
    ])

params_instance = {'batch_size': 1,
                   'shuffle': False,
                   'pin_memory': False,
                   'num_workers': 2}

for wsi, _ in inspection_dict.items():
    print(wsi)
    selected_patches_path = []
    selected_patches = []
    for key, value in inspection_dict[wsi].items():
        print(key)
        selected_patches.extend(value)
        for patch in path_patches:
            if any(item in str(patch) for item in value):
                selected_patches_path.append(patch)

    instances = Dataset_instance(selected_patches_path, transform=None, preprocess=preprocess)
    generator = DataLoader(instances, **params_instance)

    encoder.eval()
    feature_dict = {}
    with torch.no_grad():
        for i, (x_q, x_k) in enumerate(generator):

            x_q, x_k = x_q.to(device, non_blocking=True), x_k.to(device, non_blocking=True)
            q = encoder(x_q)
            q = q.squeeze().cpu().numpy()

            feature_dict[selected_patches[i]] = q

    df_matrix = pd.DataFrame(index=selected_patches, columns=selected_patches)
    combine_names = list(combinations(feature_dict, 2))
    combine_patches = list(combinations(feature_dict.values(), 2))

    for name, patch in zip(combine_names, combine_patches):
        # print(f"Similarity between {name[0]} and {name[1]}")
        similarity = cosine_similarity(patch[0], patch[1])
        df_matrix.loc[name[0], name[1]] = similarity
        df_matrix.loc[name[1], name[0]] = similarity
        # print(f"Cosine Similarity: {similarity}")
    
    df_matrix.fillna(1, inplace=True)

    sns.heatmap(df_matrix, vmin=0, vmax=1)
    plt.suptitle("Cosine Similariry")
    plt.title(wsi) 
    plt.savefig(csvdir / f"{wsi}_similarity.svg")
    plt.clf()
