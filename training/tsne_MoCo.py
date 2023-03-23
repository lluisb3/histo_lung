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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

# checkpoint = torch.load(modeldir / cfg.dataset.magnification / cfg.model.model_name / f"{experiment_name}.pt")
checkpoint = torch.load(modeldir / cfg.dataset.magnification / cfg.model.model_name / "MoCo.pt")
encoder.load_state_dict(checkpoint["encoder_state_dict"])
loss = checkpoint["loss"]
epoch = checkpoint["epoch"] + 1

print(f"Loaded encoder using as backbone {cfg.model.model_name} with a best loss of {loss} at Epoch {epoch}")

# Load patches
pyhistdir = Path(datadir / "Mask_PyHIST_v2")

dataset_path = natsorted([i for i in pyhistdir.rglob("*_densely_filtered_paths.csv")])

number_patches = 0
path_patches = []
for wsi_patches in tqdm(dataset_path, desc="Selecting patches to check model"):

    csv_instances = pd.read_csv(wsi_patches).to_numpy()
    
    path_patches.extend(csv_instances)

preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev),
        transforms.Resize(size=(model.resize_param, model.resize_param),
        antialias=True)
    ])

params_instance = {'batch_size': 1,
                   'shuffle': False,
                   'pin_memory': True,
                   'num_workers': 2}

instances = Dataset_instance(path_patches, transform=None, preprocess=preprocess)
generator = DataLoader(instances, **params_instance)

encoder.eval()
feature_matrix = np.zeros([len(path_patches), moco_dim])
with torch.no_grad():
    
    for i, (x_q, x_k) in enumerate(generator):

        x_q, x_k = x_q.to(device, non_blocking=True), x_k.to(device, non_blocking=True)

        q = encoder(x_q)
        q = q.squeeze().cpu().numpy()
        feature_matrix[i, :] = q


pca = PCA(n_components=20)
pca_features = pca.fit_transform(feature_matrix)

print(f"Explained variation per principal component: {pca.explained_variance_ratio_}")

tsne = TSNE(perplexity=30, learning_rate='auto', init='pca', verbose=1)
tsne_features = tsne.fit_transform(pca_features)

tsne_df = pd.DataFrame()

tsne_df['x'] = tsne_features[:, 0]
tsne_df['y'] = tsne_features[:, 1]
sns.scatterplot(x='x', y='y', data=tsne_df)

plt.savefig(datadir / "scaterplot.png")