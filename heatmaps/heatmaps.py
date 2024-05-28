from pathlib import Path
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from torch.utils.data import DataLoader
import openslide
import cv2 as cv
import seaborn as sns
import scipy.ndimage as ndimage
from natsort import natsorted
from training import ModelOption, yaml_load
from utils import available_magnifications
from database import Dataset_instance_MIL
import pylab
import click


thispath = Path(__file__).resolve()


def smooth_heatmap(heatmap, sigma):
    
    heatmap_smooth = ndimage.gaussian_filter(heatmap, sigma=sigma, order=0)
    
    return np.array(heatmap_smooth)


class MIL_model(torch.nn.Module):
    def __init__(self, model, hidden_space_len, cfg):

        super(MIL_model, self).__init__()
		
        self.model = model
        self.fc_input_features = self.model.input_features
        self.num_classes = self.model.num_classes
        self.hidden_space_len = hidden_space_len
        self.net = self.model.net
        self.cfg = cfg

        self.conv_layers = torch.nn.Sequential(*list(self.net.children())[:-1])

        if (torch.cuda.device_count()>1):
            # 0 para GPU buena
            self.conv_layers = torch.nn.DataParallel(self.conv_layers, device_ids=[0])


        if self.model.embedding_bool:
            if ('resnet34' in self.model.model_name):
                self.E = self.hidden_space_len
                self.L = self.E
                self.D = self.hidden_space_len
                self.K = self.num_classes

            elif ('resnet101' in self.model.model_name):
                self.E = self.hidden_space_len
                self.L = self.E
                self.D = self.hidden_space_len
                self.K = self.num_classes

            elif ('convnext' in self.model.model_name):
                self.E = self.hidden_space_len
                self.L = self.E
                self.D = self.hidden_space_len
                self.K = self.num_classes

            self.embedding = torch.nn.Linear(in_features=self.fc_input_features, out_features=self.E)
            self.post_embedding = torch.nn.Linear(in_features=self.E, out_features=self.E)

        else:
            self.fc = torch.nn.Linear(in_features=self.fc_input_features, out_features=self.num_classes)

            if ('resnet34' in self.model.model_name):
                self.L = self.fc_input_features
                self.D = self.hidden_space_len
                self.K = self.num_classes   
            
            elif ('resnet101' in self.model.model_name):
                self.L = self.E
                self.D = self.hidden_space_len
                self.K = self.num_classes

            elif ('convnext' in self.model.model_name):
                self.L = self.E
                self.D = self.hidden_space_len
                self.K = self.num_classes
		
        if (self.model.pool_algorithm=="attention"):
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(self.L, self.D),
                torch.nn.Tanh(),
                torch.nn.Linear(self.D, self.K)
            )
            
            if "NoChannel" in self.cfg.data_augmentation.featuresdir:
                # print("== Attention No Channel ==")
                self.embedding_before_fc = torch.nn.Linear(self.E * self.K, self.E)

            elif "AChannel" in self.cfg.data_augmentation.featuresdir:
                # print("== Attention with A Channel for multilabel ==")
                self.attention_channel = torch.nn.Sequential(torch.nn.Linear(self.L, self.D),
                                                    torch.nn.Tanh(),
                                                    torch.nn.Linear(self.D, 1))
                self.embedding_before_fc = torch.nn.Linear(self.E, self.E)

            

        self.embedding_fc = torch.nn.Linear(self.E, self.K)

        self.dropout = torch.nn.Dropout(p=self.model.dropout)
        # self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        self.LayerNorm = torch.nn.LayerNorm(self.E * self.K, eps=1e-5)
        # self.activation = self.tanh
        self.activation = self.relu 

    def forward(self, x, conv_layers_out):

        #if used attention pooling
        A = None
        #m = torch.nn.Softmax(dim=1)

        if x is not None:
            #print(x.shape)
            conv_layers_out=self.conv_layers(x)
            #print(x.shape)

            conv_layers_out = conv_layers_out.view(-1, self.fc_input_features)

        if self.model.embedding_bool:
            embedding_layer = self.embedding(conv_layers_out)
							
            #embedding_layer = self.LayerNorm(embedding_layer)
            features_to_return = embedding_layer
            embedding_layer = self.dropout(embedding_layer)

        else:
            embedding_layer = conv_layers_out
            features_to_return = embedding_layer


        A = self.attention(features_to_return)

        A = torch.transpose(A, 1, 0)

        A = F.softmax(A, dim=1)

        wsi_embedding = torch.mm(A, features_to_return)

        if "NoChannel" in self.cfg.data_augmentation.featuresdir:
            # print("== Attention No Channel ==")
            wsi_embedding = wsi_embedding.view(-1, self.E * self.K)

            cls_img = self.embedding_before_fc(wsi_embedding)

        elif "AChannel" in self.cfg.data_augmentation.featuresdir:
            # print("== Attention with A Channel for multilabel ==")
            attention_channel = self.attention_channel(wsi_embedding)

            attention_channel = torch.transpose(attention_channel, 1, 0)

            attention_channel = F.softmax(attention_channel, dim=1)

            cls_img = torch.mm(attention_channel, wsi_embedding)

            # cls_img = self.embedding_before_fc(cls_img)

        cls_img = self.activation(cls_img)

        cls_img = self.dropout(cls_img)

        Y_prob = self.embedding_fc(cls_img)

        Y_prob = torch.squeeze(Y_prob)

        return Y_prob, A
    
    
experiment_name = "f_MIL_res34v2_v2_rumc_best_cosine_v3_fold_0"
downsample_factor = 4


experiment_name = "f_MIL_res34v2_v2_rumc_best_cosine_v3_fold_0"
downsample_factor = 4


@click.command()
@click.option(
    "--wsi_name",
    default="TCGA-18-3417-01Z-00-DX1.tif",
    prompt="Name of the WSI to perform study",
    help="Name of the WSI to perform study",
)
@click.option(
    "--sigma",
    default=8,
    prompt="Value of sigma applied to the gaussian filter",
    help="Value of sigma applied to the gaussian filter",
)
def main(wsi_name, sigma):

    datadir = Path(thispath.parent.parent / "data" / "tcga")
    tcgadir = Path(Path(datadir) / "wsi")
    patchdir = Path(Path(datadir) / "patches")
    maskdir = Path(Path(datadir) / "mask")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Seed for reproducibility
    seed = 33
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # list_wsi = natsorted([f.name for f in Path(patchdir).iterdir() if f.is_dir()], key=str)

    # list_wsi = [s[:-4] for s in list_wsi]

    # if wsi_name in list_wsi:
    #     wsi_id = list_wsi.index(wsi_name)
    #     print(f"The index of '{wsi_name}' is: {wsi_id}")
    # else:
    #     print(f"'{wsi_name}' is not in the list.")

    labels_df = pd.read_csv(f"{datadir}/labels_tcga_all.csv", index_col=0)

    # selected_columns = ['cancer_nscc_adeno', 'cancer_nscc_squamous']
    # filtered_df = labels_df.loc[wsi_name, selected_columns]

    # Create the list of tuples
    # select_wsi = []

    # for index, row in filtered_df.iterrows():
    #     if row['cancer_nscc_adeno'] == 1:
    #         select_wsi.append((index, "LUAD"))
    #     elif row['cancer_nscc_squamous'] == 1:
    #         select_wsi.append((index, "LUSC"))

    for index, row in labels_df.iterrows():
        if index == wsi_name[:-4]:
            if row['cancer_nscc_adeno'] == 1:
                groundtruth = "LUAD"
            elif row['cancer_nscc_squamous'] == 1:
                groundtruth = "LUSC"

    # sample = select_wsi[0]
    print(f"Heatmap for sample: {wsi_name} with Groundtruth label: {groundtruth}")

    # tif_dir = natsorted([i for i in tcgadir.rglob("*.tif")], key=str)
    # print(tif_dir)
    
    # for dir in tif_dir:
        # if sample[0] in str(dir):
        #         file_path = dir
        #         print(f"Loaded WSI from {file_path}")
        #         break

    modeldir = Path(thispath.parent.parent / "trained_models" / "MIL" / "f_MIL_res34v2_v2_rumc_best_cosine_v3")

    cfg = yaml_load(modeldir / f"config_f_MIL_res34v2_v2_rumc_best_cosine_v3.yml")

    checkpoint = torch.load(modeldir / "fold_0" / "checkpoint.pt")

    print(f"Loaded Model using {cfg.model.model_name} as backbone")
          
    model = ModelOption(cfg.model.model_name,
                    cfg.model.num_classes,
                    freeze=cfg.model.freeze_weights,
                    num_freezed_layers=cfg.model.num_frozen_layers,
                    dropout=cfg.model.dropout,
                    embedding_bool=cfg.model.embedding_bool,
                    pool_algorithm=cfg.model.pool_algorithm
                    )

    hidden_space_len = cfg.model.hidden_space_len

    net = MIL_model(model, hidden_space_len, cfg)

    net.load_state_dict(checkpoint["model_state_dict"], strict=False)
    net.to(device)
    net.eval()

    preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev),
                transforms.Resize(size=(model.resize_param, model.resize_param),
                antialias=True)
        ])
    
    file = openslide.open_slide(f"{tcgadir}/{wsi_name}")
    # mpp = file.properties['openslide.mpp-x']

    # level_downsamples = file.level_downsamples
    # mags = available_magnifications(mpp, level_downsamples)

    mask = cv.imread(str(maskdir / f"{wsi_name}" / f"{wsi_name}_mask_use.png"))
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)

    print(f"Mask shape: {mask.shape}")

    # mask_np = cv.resize(mask, (int(mask.shape[1]/downsample_factor), int(mask.shape[0]/downsample_factor)))

    thumb = file.get_thumbnail((mask.shape[1], mask.shape[0]))

    mask_empty = np.zeros((mask.shape[0], mask.shape[1]))
    
    sampledir = Path(patchdir / f"{wsi_name}")
    metadata_preds = pd.read_csv(sampledir / f"{wsi_name}_coords_densely.csv", header=None)
    # patches = pd.read_csv(sampledir / f"{sample[0]}.tif_paths_densely.csv", header=None).values
    patches = natsorted([[str(i)] for i in sampledir.rglob("*.png")], key=str)

    names = metadata_preds.iloc[:, 0]
    coords_x = metadata_preds.iloc[:, 3].values
    coords_y = metadata_preds.iloc[:, 2].values

    #params generator instances
    batch_size_instance = 1
    num_workers = 2

    params_instance = {'batch_size': batch_size_instance,
            #'shuffle': True,
            'num_workers': num_workers}

    instances = Dataset_instance_MIL(patches, preprocess=preprocess)
    validation_generator_instance = DataLoader(instances, **params_instance)

    downsample_factor = 32
    n_elems = len(patches)
    dicts = []
    features = [] 
    with torch.no_grad():
        for i, patch in enumerate(validation_generator_instance):
            patch = patch.to(device, non_blocking=True)
            
            coord_x = int((coords_x[i])/downsample_factor)
            coord_y = int((coords_y[i])/downsample_factor)

            # forward + backward + optimize
            feats = net.conv_layers(patch)
            feats = feats.view(-1, net.fc_input_features)
            feats_np = feats.cpu().data.numpy()

            features.extend(feats_np)
            
            
            d = {
            "ID": names[i],
            "coord_x": int(coord_x),
            "coord_y": int(coord_y),
            "prob":None  
            }
            dicts.append(d)

    features_np = np.reshape(features,(n_elems, net.fc_input_features))
    #torch.cuda.empty_cache()

    net.zero_grad()

    inputs = torch.tensor(features_np).float().to(device,non_blocking=True)

    pred_wsi, attention_weights = net(None, inputs)

    #attention_weights = torch.transpose(attention_weights, 1, 0)

    pred_wsi = pred_wsi.cpu().data.numpy()
    print("SCLC    LUAD     LUSC     Normal")
    print(f"Prediction score: {pred_wsi}")

    final_prediction_id = pred_wsi.argmax()
    my_cmap_red = sns.color_palette("Reds", 255, as_cmap=True)
    my_cmap_green = sns.color_palette("Greens", 255, as_cmap=True)
    my_cmap_yellow = sns.color_palette("YlOrBr", 255, as_cmap=True)
    my_cmap_blue = sns.color_palette("Blues", 255, as_cmap=True)

    if final_prediction_id == 0:
        final_prediction = "SCLC"
        my_cmap = my_cmap_yellow

    elif final_prediction_id == 1:
        final_prediction = "LUAD"
        my_cmap = my_cmap_green

    elif final_prediction_id == 2:
        final_prediction = "LUSC"
        my_cmap = my_cmap_red

    elif final_prediction_id == 3:
        final_prediction = "Normal"
        my_cmap = my_cmap_blue
    
    outputdir = Path(datadir.parent / "outputs")
    Path(outputdir).mkdir(exist_ok=True, parents=True)

    File = {'filename': [wsi_name],
            'prediction': [final_prediction], 
            'groundtruth': [groundtruth]}

    df_prediction = pd.DataFrame.from_dict(File)
                  
    filename_prediction = Path(outputdir / "predictions.csv")
    df_prediction.to_csv(filename_prediction) 
    
    print(f"=== Final prediction of the model: {final_prediction} ===")
    print(f"=== Groundtruth: {groundtruth} ===")


    attentions_np = attention_weights.cpu().data.numpy()

    for i in range(attentions_np.shape[1]):
        #dicts[i]["prob"]=(outputs_np[i][class_colon]*attention_weights[i][class_colon])
        dicts[i]["prob"] = attentions_np[final_prediction_id, i]

        #filename_save_mask = '/home/niccolo/ExamodePipeline/Multiple_Instance_Learning/Colon/images/heat_map_'+wsi+'_'+type_tissue+'_ss_'+MAGNIFICATION+'.png'        


    torch.cuda.empty_cache()

    pixel_size = int(224/(downsample_factor))

    for d in dicts:
        x_cord_m = d['coord_x']
        y_cord_m = d['coord_y']
        
        x_cord_f = x_cord_m+pixel_size
        y_cord_f = y_cord_m+pixel_size
        for x in range(x_cord_m, x_cord_f):
            for y in range(y_cord_m, y_cord_f):
                mask_empty[x,y]=d['prob']

    mask_copy = mask_empty

    heatmap_np = np.uint8(mask_copy*600)

    heatmap_smooth_np = smooth_heatmap(heatmap_np, sigma)
    # heatmap_smooth_np[heatmap_smooth_np < 0.000002] = 0


    Fi = pylab.gcf()
    DefaultSize = Fi.get_size_inches()

    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(1600.0/float(DPI),1200.0/float(DPI))

    plt.clf()
    plt.imshow(thumb)
    plt.imshow(15*heatmap_smooth_np, alpha=0.7, cmap=my_cmap)
    plt.savefig(outputdir / f"heatmap_{wsi_name[:-4]}.png")

    print(f"Heatmap saved on {outputdir}")


if __name__ == '__main__':
    main()
