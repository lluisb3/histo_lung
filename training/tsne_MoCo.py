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
import time
from utils import timer
import umap
import umap.plot


thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Load PyTorch model
experiment_name = "MoCo_resnet101_scheduler_5"

modeldir = Path(thispath.parent.parent / "trained_models" / "MoCo" / experiment_name)

cfg = yaml_load(modeldir / f"config_{experiment_name}.yml")

model = ModelOption(cfg.model.model_name,
                cfg.model.num_classes,
                freeze=cfg.model.freeze_weights,
                dropout=cfg.model.dropout,
                embedding_bool=cfg.model.embedding_bool
                )    

# Encoder and momentum encoder
moco_dim = cfg.training.moco_dim

encoder = Encoder(model, dim=moco_dim).to(device)

checkpoint = torch.load(modeldir / cfg.dataset.magnification / cfg.model.model_name / f"{experiment_name}.pt")
# checkpoint = torch.load(modeldir / cfg.dataset.magnification / cfg.model.model_name / "MoCo.pt")
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

path_patches = path_patches[:4000]
preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev),
        transforms.Resize(size=(model.resize_param, model.resize_param),
        antialias=True)
    ])

params_instance = {'batch_size': 128,
                   'shuffle': False,
                   'pin_memory': True,
                   'num_workers': 2}

instances = Dataset_instance(path_patches, transform=None, preprocess=preprocess)
generator = DataLoader(instances, **params_instance)

encoder.eval()

start = time.time()
feature_matrix = np.zeros([len(path_patches), moco_dim], dtype=float)
with torch.no_grad():
    
    for i, (x_q, _) in tqdm(enumerate(generator)):
        x_q = x_q.to(device, non_blocking=True)
        
        q = encoder(x_q)
        q = q.squeeze().cpu().numpy()

        if len(q) < params_instance["batch_size"]:
            feature_matrix[len(path_patches)-len(q):len(path_patches), :] = q
        else:
            feature_matrix[i*len(q):(i+1)*len(q), :] = q
            
message = timer(start, time.time())
print(f"Time to build feature matrix {message}")
start = time.time()

pca = PCA(n_components=20)
pca_features = pca.fit_transform(feature_matrix)

print(f"Explained variation per principal component: {pca.explained_variance_ratio_}")

# tsne = TSNE(perplexity=30, learning_rate='auto', init='pca', verbose=1)
# tsne_features = tsne.fit_transform(pca_features)

# tsne_df = pd.DataFrame()

# tsne_df['x'] = tsne_features[:, 0]
# tsne_df['y'] = tsne_features[:, 1]
# sns.scatterplot(x='x', y='y', data=tsne_df)

# plt.savefig(datadir / "scaterplot.png")
plt.figure()
mapper = umap.UMAP().fit(pca_features)
umap.plot.points(mapper)
plt.savefig(datadir / "uMap.png")


message = timer(start, time.time())
print(f"Time perform uMap {message}")
