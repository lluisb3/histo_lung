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
from sklearn.decomposition import PCA
import seaborn as sns
import umap
import umap.plot
import click

thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.option(
    "--experiment_name",
    default="MoCo_resnet101_scheduler_5",
    prompt="Name of the MoCo experiment name to compute similarity metrics",
    help="Name of the MoCo experiment name to compute similarity metrics",
)
@click.option(
    "--version_patch_similarity",
    default="v1",
    prompt="Version for the patch similarity selection 'v1' or 'v2'",
    help="Version for the patch similarity selection 'v1' or 'v2'",
)
def main(experiment_name, version_patch_similarity):
# Load PyTorch model

    seed = 33
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

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

    checkpoint = torch.load(modeldir / cfg.dataset.magnification / cfg.model.model_name / f"{experiment_name}.pt")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    loss = checkpoint["loss"]
    epoch = checkpoint["epoch"] + 1

    print(f"Loaded encoder using as backbone {cfg.model.model_name} with a best loss of {loss} at Epoch {epoch}")

    # Load patches
    pyhistdir = Path(datadir / "Mask_PyHIST_v2")

    if version_patch_similarity == "v1":
        similaridir = Path(thispath.parent.parent / "patch_similarity" / "patch_similarity_images")

    elif version_patch_similarity == "v2":
        similaridir = Path(thispath.parent.parent / "patch_similarity" / "patch_similarity_images_v2")
    else:
        print("Error: wrong version selected")

    dataset_path = natsorted([i for i in pyhistdir.rglob("*_densely_filtered_paths.csv")
                              if "000030303300314205" in str(i)
                              or "000030494900323685" in str(i)
                              or "000030689500332896" in str(i)
                              or "000030734200335036" in str(i)
                              or "000030734200335038" in str(i)
                              or "000030734600335056" in str(i)
                              or "000033382400482121" in str(i)
                              or "000030689200333236" in str(i)
                              or "000030769200337772" in str(i)
                              or "000030961000347399" in str(i)
                              or "000031634700383219" in str(i)
                              or "000030737800334725" in str(i)
                              or "000031634700383219" in str(i)])

    selected_patches = natsorted([e for e in similaridir.rglob("*.png")])

    cells = []
    glands = []
    stroma = []
    for patch in selected_patches:
        if patch.parent.stem == "cells":
            cells.append(patch.stem)
        elif patch.parent.stem == "glands":
            glands.append(patch.stem)
        elif patch.parent.stem == "stroma":
            stroma.append(patch.stem)

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
                    'pin_memory': False,
                    'num_workers': 2}

    selected_patches_path = []
    labels = []
    for wsi in tqdm(path_patches):
        if Path(wsi[0]).stem in cells:
            selected_patches_path.append(wsi)
            label = "cells"
            labels = np.append(labels, label)
        elif Path(wsi[0]).stem in glands:
            selected_patches_path.append(wsi)
            label = "glands"
            labels = np.append(labels, label)
        elif Path(wsi[0]).stem in stroma:
            selected_patches_path.append(wsi)
            label = "stroma"
            labels = np.append(labels, label)

    selected_patches_path.sort()
    # print(selected_patches_path)
    instances = Dataset_instance(selected_patches_path, transform=None, preprocess=preprocess)
    generator = DataLoader(instances, **params_instance)

    encoder.eval()
    feature_dict = {}
    feature_matrix = []
    with torch.no_grad():
        for i, (x_q, x_k) in enumerate(generator):

            x_q, x_k = x_q.to(device, non_blocking=True), x_k.to(device, non_blocking=True)
            q = encoder(x_q)
            q = q.squeeze().cpu().numpy()

            feature_dict[Path(str(selected_patches_path[i])).stem] = q
            feature_matrix = np.append(feature_matrix, q)

    # uMap
    feature_matrix = feature_matrix.reshape(len(selected_patches_path), moco_dim)
    pca = PCA(n_components=20)
    pca_features = pca.fit_transform(feature_matrix)
    print(f"Explained variation per principal component: {pca.explained_variance_ratio_}")

    # tsne = TSNE(perplexity=30, learning_rate='auto', init='pca', verbose=1)
    # tsne_features = tsne.fit_transform(pca_features)

    # tsne_df = pd.DataFrame()

    # tsne_df['x'] = tsne_features[:, 0]
    # tsne_df['y'] = tsne_features[:, 1]
    # sns.scatterplot(x='x', y='y', data=tsne_df)

    # plt.savefig(datadir / "scaterplot_similarity.png")
    plt.figure()
    mapper = umap.UMAP().fit(pca_features)
    umap.plot.points(mapper, labels=labels, theme='red')
    plt.savefig(similaridir / f"{experiment_name}_uMap_similarity_v2.svg")

if __name__ == '__main__':
    main()
