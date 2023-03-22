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
import matplotlib.pyplot as plt
from itertools import combinations
from easydict import EasyDict as edict

thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load PyTorch model
experiment_name = "MoCo_try_Adam"

modeldir = Path(thispath.parent.parent / "trained_models" / experiment_name)

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

optimizer = getattr(torch.optim, cfg.training.optimizer)
optimizer = optimizer(encoder.parameters(), **cfg.training.optimizer_args)

# checkpoint = torch.load(modeldir / cfg.dataset.magnification / cfg.model.model_name / f"{experiment_name}.pt")
checkpoint = torch.load(modeldir / cfg.dataset.magnification / cfg.model.model_name / "MoCo.pt")
encoder.load_state_dict(checkpoint["encoder_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
loss = checkpoint["loss"]
epoch = checkpoint["epoch"] + 1

print(f"Loaded encoder using as backbone {cfg.model.model_name} with a best loss of {loss} at Epoch {epoch}")

# Load patches
pyhistdir = Path(datadir / "Mask_PyHIST_v2")

selected_wsi = ["000030303300314205", "000030494900323685"]

dataset_path = natsorted([i for i in pyhistdir.rglob("*_densely_filtered_paths.csv") if selected_wsi[0] in str(i) or selected_wsi[1] in str(i)])

number_patches = 0
path_patches = []
for wsi_patches in tqdm(dataset_path, desc="Selecting patches to check model"):

    csv_instances = pd.read_csv(wsi_patches).to_numpy()
    
    path_patches.extend(csv_instances)

error_bw_patches = ["000030303300314205_06210",
                    "000030303300314205_06378",
                    "000030303300314205_07218",
                    "000030303300314205_07050"]

cell_patches = ["000030303300314205_05076",
                "000030303300314205_05078",
                "000030303300314205_05077"]

cell_2sizes_patches = ["000030303300314205_07085",
                       "000030303300314205_07092",
                       "000030303300314205_07091"]

stroma_patches = ["000030303300314205_06422",
                  "000030303300314205_06423",
                  "000030303300314205_06444",
                  "000030303300314205_06592"]

cells_gaps_patches = ["000030303300314205_08750",
                      "000030303300314205_08741",
                      "000030303300314205_08742",
                      "000030303300314205_08753"]

black_thinguis_patches = ["000030303300314205_09449",
                          "000030303300314205_09453",
                          "000030303300314205_09585",
                          "000030303300314205_09617",
                          "000030303300314205_09616"]

red_gland_patches = ["000030494900323685_02127",
                     "000030494900323685_02770",
                     "000030494900323685_02513"]

lipids_cells_patches = ["000030494900323685_04173",
                        "000030494900323685_05220"]

lipids_stroma_patches = ["000030494900323685_05275",
                         "000030494900323685_05404",
                         "000030494900323685_05340",
                         "000030494900323685_04240"]

lipids_glands_patches = ["000030494900323685_02154",
                         "000030494900323685_02770",
                         "000030494900323685_02642"]

inspection_dict = edict({"error_bw_patches": error_bw_patches,
                         "cell_patches": cell_patches,
                         "cell_2sizes_patches": cell_2sizes_patches,
                         "stroma_patches": stroma_patches,
                         "cells_gaps_patches": cells_gaps_patches,
                         "black_thinguis_patches": black_thinguis_patches,
                         "red_gland_patches": red_gland_patches,
                         "lipids_cells_patches": lipids_cells_patches,
                         "lipids_stroma_patches": lipids_stroma_patches,
                         "lipids_glands_patches": lipids_glands_patches})


selected_patches = inspection_dict.error_bw_patches + inspection_dict.cell_patches
print(selected_patches)
selected_patches_path = []
for patch in path_patches:
    if any(item in str(patch) for item in selected_patches):
        selected_patches_path.append(patch)

preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev),
        transforms.Resize(size=(model.resize_param, model.resize_param))
    ])

params_instance = {'batch_size': 1,
                   'shuffle': False,
                   'pin_memory': True,
                   'num_workers': 2}

instances = Dataset_instance(selected_patches_path, transform=None, preprocess=preprocess)
generator = DataLoader(instances, **params_instance)

encoder.eval()
feature_matrix = np.zeros([len(selected_patches_path), moco_dim])
feature_dict = {}
with torch.no_grad():
    
    for i, (x_q, x_k) in enumerate(generator):

        x_q, x_k = x_q.to(device, non_blocking=True), x_k.to(device, non_blocking=True)

        q = encoder(x_q)
        q = q.squeeze().cpu().numpy()
        feature_matrix[i, :] = q
        feature_dict[selected_patches[i]] = q

combine_names = combinations(feature_dict, 2)
combine_patches = combinations(feature_dict.values(), 2)
for name, patch in zip(combine_names, combine_patches):
    print(f"Similarity between {name[0]} and {name[1]}")
    cosine_similarity(patch[0], patch[1])
