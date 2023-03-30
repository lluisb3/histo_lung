from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import albumentations as A
import time
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms
from sklearn import metrics 
import os
import argparse
from natsort import natsorted
from transformers import BertModel, BertPreTrainedModel, RobertaConfig, RobertaModel
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
#from transformers.modeling_longformer import LongformerSelfAttention
from transformers import LongformerConfig, LongformerModel, LongformerSelfAttention
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification, AutoTokenizer

from ast import literal_eval
from database import Dataset_instance, Dataset_instance_MIL, Dataset_bag_MIL
from training.mil import MIL_model
from training.models import ModelOption
from training.encoder import Encoder
from training.utils_trainig import momentum_step, contrastive_loss, update_queue
from training.utils_trainig import yaml_load, initialize_wandb, edict2dict
from utils import timer
import wandb
import logging
import click

thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# torch.backends.cudnn.benchmark = True


def select_parameters_colour():
    hue_min = -15
    hue_max = 8

    sat_min = -20
    sat_max = 10

    val_min = -8
    val_max = 8


    p1 = np.random.uniform(hue_min,hue_max,1)
    p2 = np.random.uniform(sat_min,sat_max,1)
    p3 = np.random.uniform(val_min,val_max,1)

    return p1[0],p2[0],p3[0]

def select_rgb_shift():
    r_min = -10
    r_max = 10

    g_min = -10
    g_max = 10

    b_min = -10
    b_max = 10


    p1 = np.random.uniform(r_min,r_max,1)
    p2 = np.random.uniform(g_min,g_max,1)
    p3 = np.random.uniform(b_min,b_max,1)

    return p1[0],p2[0],p3[0]

def select_elastic_distorsion():
    sigma_min = 0
    sigma_max = 20

    alpha_affine_min = -20
    alpha_affine_max = 20

    p1 = np.random.uniform(sigma_min,sigma_max,1)
    p2 = np.random.uniform(alpha_affine_min,alpha_affine_max,1)

    return p1[0],p2[0]

def select_scale_distorsion():
    scale_min = 0.8
    scale_max = 1.0

    p1 = np.random.uniform(scale_min,scale_max,1)

    return p1[0]

def select_grid_distorsion():
    dist_min = 0
    dist_max = 0.2

    p1 = np.random.uniform(dist_min,dist_max,1)

    return p1[0]

def generate_transformer(label, prob = 0.5):
    list_operations = []
    probas = np.random.rand(7)

    if (probas[0]>prob):
        #print("VerticalFlip")
        list_operations.append(A.VerticalFlip(always_apply=True))
    if (probas[1]>prob):
        #print("HorizontalFlip")
        list_operations.append(A.HorizontalFlip(always_apply=True))
    #"""
    if (probas[2]>prob):
        #print("RandomRotate90")
        #list_operations.append(A.RandomRotate90(always_apply=True))

        p_rot = np.random.rand(1)[0]
        if (p_rot<=0.33):
            lim_rot = 90
        elif (p_rot>0.33 and p_rot<=0.66):
            lim_rot = 180
        else:
            lim_rot = 270
        list_operations.append(A.SafeRotate(always_apply=True, limit=(lim_rot,lim_rot+1e-4), interpolation=1, border_mode=4))

    if (probas[3]>prob):
        #print("HueSaturationValue")
        p1, p2, p3 = select_parameters_colour()
        list_operations.append(A.HueSaturationValue(always_apply=True,hue_shift_limit=(p1,p1+1e-4),sat_shift_limit=(p2,p2+1e-4),val_shift_limit=(p3,p3+1e-4)))

    if (probas[4]>prob):
        p1 = select_scale_distorsion()
        list_operations.append(A.RandomResizedCrop(height=224, width=224, scale=(p1,p1+1e-4), always_apply=True))
        #print(p1,p2,p3)

    if (probas[5]>prob):
        #p1, p2 = select_elastic_distorsion()
        list_operations.append(A.ElasticTransform(alpha=1,border_mode=4, sigma=5, alpha_affine=5,always_apply=True))
        #print(p1,p2)

    if (probas[6]>prob):
        p1 = select_grid_distorsion()
        list_operations.append(A.GridDistortion(num_steps=3, distort_limit=p1, interpolation=1, border_mode=4, always_apply=True))
        #print(p1)
    pipeline_transform = A.Compose(list_operations)

    return pipeline_transform


def get_generator_instances(csv_instances, preprocess, batch_size, pipeline_transform, num_workers):

    params_instance = {'batch_size': batch_size,
                    'num_workers': num_workers,
                    'pin_memory': True,
                    'shuffle': True}

    instances = Dataset_instance(csv_instances, pipeline_transform, preprocess)
    generator = DataLoader(instances, **params_instance)

    return generator


def accuracy_micro(y_true, y_pred):

    y_true_flatten = y_true.flatten()
    y_pred_flatten = y_pred.flatten()
    
    return metrics.accuracy_score(y_true_flatten, y_pred_flatten)

    
def accuracy_macro(y_true, y_pred):

    n_classes = len(y_true[0])

    acc_tot = 0.0

    for i in range(n_classes):

        acc = metrics.accuracy_score(y_true[i,:], y_pred[i,:])
        #print(acc)
        acc_tot = acc_tot + acc

    acc_tot = acc_tot/n_classes

    return acc_tot


def evaluate_validation_set(net, criterion, generator, features_valid):
    #accumulator for validation set
    y_pred = []
    y_true = []

    validation_loss = 0.0

    net.eval()

    with torch.no_grad():
        for i, (wsi_id, labels) in enumerate(generator):

            label_wsi = labels[0].cpu().numpy().flatten()
    
            labels_local = labels.float().flatten().to(device, non_blocking=True)

            #read csv with instances
            # n_indices = len(features_np)
            # #indices = np.random.choice(n_indices,n_indices,replace=False)
            # indices = np.random.permutation(n_indices)
            # features_np = features_np[indices]
            features_wsi = features_valid.loc[wsi_id].values
            inputs = torch.as_tensor(features_wsi).float().to(device, non_blocking=True)

            logits_img, _ = net(None, inputs)
            
            #loss img
            loss_img = criterion(logits_img, labels_local)

            sigmoid_output_img = F.sigmoid(logits_img)
            outputs_wsi_np_img = sigmoid_output_img.cpu().data.numpy()
            validation_loss = validation_loss + ((1 / (i+1)) * (loss_img.item() - validation_loss))
            
            logging.info("pred_img: " + str(outputs_wsi_np_img))
            logging.info("validation_loss: " + str(validation_loss))

            output_norm = np.where(outputs_wsi_np_img > 0.5, 1, 0)

            y_pred = np.append(y_pred,output_norm)
            y_true = np.append(y_true,label_wsi)

            micro_accuracy_train = accuracy_micro(y_true, y_pred)
            logging.info("micro_accuracy " + str(micro_accuracy_train))    
    
    return validation_loss, micro_accuracy_train


@click.command()
@click.option(
    "--config_file",
    default="config_MIL",
    prompt="Name of the config file without extension",
    help="Name of the config file without extension",
)
@click.option(
    "--exp_name_moco",
    default="MoCo_try_cpu_256_resnet34",
    prompt="Name of the MoCo experiment",
    help="Name of the MoCo experiment",
)
def main(config_file, exp_name_moco):
    # Read the configuration file
    configdir = Path(thispath.parent / f"{config_file}.yml")
    cfg = yaml_load(configdir)

    # Create directory to save the resuls
    outputdir = Path(thispath.parent.parent / "trained_models" / "MIL" / f"{cfg.experiment_name}")
    Path(outputdir).mkdir(exist_ok=True, parents=True)

    # wandb login
    # wandb.login()

    # if cfg.wandb.enable:
    #         # key = os.environ.get("WANDB_API_KEY")
    #         wandb_run = initialize_wandb(cfg, outputdir)
    #         wandb_run.define_metric("epoch", summary="max")

    # For logging
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                        encoding='utf-8',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(outputdir / "debug.log"),
                            logging.StreamHandler()
                        ],
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    
    logging.info(f"CUDA current device {torch.device('cuda:0')}")
    logging.info(f"CUDA devices available {torch.cuda.device_count()}")

    # Seed for reproducibility
    seed = 33
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # MoCo model maybe not necesary if lodaded features directly
    # #path model file
    # experiment_name = "MoCo_try_Adam"

    # mocodir = Path(thispath.parent.parent / 
    #                "trained_models" / 
    #                "MoCo" / 
    #                experiment_name)

    # cfg_moco = yaml_load(mocodir / "config_{experiment_name}.yml")

    # model_moco = ModelOption(cfg.model.model_name,
    #                 cfg.model.num_classes,
    #                 freeze=cfg.model.freeze_weights,
    #                 num_freezed_layers=cfg.model.num_frozen_layers,
    #                 dropout=cfg.model.dropout,
    #                 embedding_bool=cfg.model.embedding_bool
    #                 )    

    # # Encoder and momentum encoder
    # moco_dim = cfg_moco.training.moco_dim

    # encoder = Encoder(model_moco, dim=moco_dim).to(device)

    # checkpoint_moco = torch.load(Path(mocodir /
    #                              cfg_moco.dataset.magnification / 
    #                              cfg_moco.model.model_name / 
    #                              "MoCo.pt"))
    # encoder.load_state_dict(checkpoint_moco["encoder_state_dict"])
    # loss_moco = checkpoint_moco["loss"]
    # epoch_moco = checkpoint_moco["epoch"] + 1

    # print(f"Loaded encoder using as backbone {model_moco.model_name} with a best"
    #       f"loss of {loss_moco} at Epoch {epoch_moco}")

    # preprocess_moco = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=cfg_moco.dataset.mean, std=cfg_moco.dataset.stddev),
    #         transforms.Resize(size=(model_moco.resize_param, model_moco.resize_param),
    #         antialias=True)
    #     ])

    # params_dataloader_moco = {'batch_size': 1,
    #                           'shuffle': False,
    #                           'pin_memory': True,
    #                           'num_workers': 2}

    # patches_dataset = Dataset_instance(patches_path, transform=None, preprocess=preprocess_moco)
    # generator = DataLoader(patches_dataset, **params_dataloader_moco)

    # encoder.eval()
    # feature_patches_dict = {}
    # with torch.no_grad():
        
    #     for i, (x_q, x_k) in enumerate(generator):

    #         x_q, x_k = x_q.to(device, non_blocking=True), x_k.to(device, non_blocking=True)

    #         q = encoder(x_q)
    #         q = q.squeeze().cpu().numpy()
    #         feature_patches_dict[patches_path.stem[i]] = q

    # Loading Data Spit
    k = 10

    data_split = pd.read_csv(Path(datadir / f"{k}_fold_crossvalidation_data_split.csv"), index_col=0)
    train_dataset_k = []
    validation_dataset_k = []
    train_labels_k = []
    validation_labels_k = []

    for fold, _ in data_split.iterrows():
        train_wsi = literal_eval(data_split.loc[fold]["images_train"])
        validation_wsi = literal_eval(data_split.loc[fold]["images_test"])
        labels_train = literal_eval(data_split.loc[fold]["labels_train"])
        labels_validation = literal_eval(data_split.loc[fold]["labels_test"])
        train_dataset_k.append(train_wsi)
        validation_dataset_k.append(validation_wsi)
        train_labels_k.append(labels_train)
        validation_labels_k.append(labels_validation)

    # Load fold 0
    train_dataset = train_dataset_k[0]
    validation_dataset = validation_dataset_k[0]
    train_labels = train_labels_k[0]
    validation_labels = validation_labels_k[0]

    test_csv = pd.read_csv(Path(datadir / f"labels_test.csv"), index_col=0)
    test_dataset = test_csv.index
    test_dataset = [i.replace("/", "-") for i in test_dataset]
    test_labels = test_csv.values

    # Load patches path
    pyhistdir = Path(datadir / "Mask_PyHIST_v2")

    dataset_path = natsorted([i for i in pyhistdir.rglob("*_densely_filtered_paths.csv")])

    patches_path = {}
    patches_names = {}
    for wsi_patches in tqdm(dataset_path, desc="Selecting patches to extract features"):

        csv_instances = pd.read_csv(wsi_patches).to_numpy()
        
        name = wsi_patches.parent.stem
        patches_path[name] = csv_instances
        patches_names[name] = []

        for instance in csv_instances:
                patches_names[name].append(str(instance).split("/")[-1])

    logging.info(f"Total number of patches for train/validation/test {len(patches_path)}")

    patches_train = {}
    patches_validation = []
    patches_test = []
    for value, key in zip(patches_names.values(), patches_names.keys()):
        if key in train_dataset:
            patches_train[key] = value
        if key in validation_dataset:
            patches_validation.extend(value)
        if key in test_dataset:
            patches_test.extend(value)

    logging.info(f"Total number of patches for train {len(patches_train)}")
    logging.info(f"Total number of patches for validation {len(patches_validation)}")
    logging.info(f"Total number of patches for test {len(patches_test)}")

    # Load datasets
    batch_size_bag = cfg.dataloader.batch_size_bag


    params_train_bag = {'batch_size': batch_size_bag,
                        'shuffle': True}

    params_valid_bag = {'batch_size': len(validation_dataset),
                        'shuffle': False}

    training_set_bag = Dataset_bag_MIL(train_dataset, train_labels)
    training_generator_bag = DataLoader(training_set_bag, **params_train_bag)

    validation_set_bag = Dataset_bag_MIL(validation_dataset, validation_labels)
    validation_generator_bag = DataLoader(validation_set_bag, **params_valid_bag)

    # Load features from MoCo model
    experiment_name = exp_name_moco

    logging.info(f"== Loading features from {experiment_name} ==")

    mocodir = Path(thispath.parent.parent / 
                "trained_models" / 
                "MoCo" / 
                experiment_name)

    cfg_moco = yaml_load(mocodir / f"config_{experiment_name}.yml")

    checkpoint_moco = torch.load(Path(mocodir /
                                cfg_moco.dataset.magnification / 
                                cfg_moco.model.model_name / 
                                f"{exp_name_moco}.pt"))

    # df_features = pd.read_csv(mocodir / f"features_{experiment_name}.csv", index_col=0)

    # features_valid = df_features.loc[patches_validation]
    # features_test = df_features.loc[patches_test]

    # Initialize Bert Tokenizer
    logging.info("== Initialize BERT ==")
    bert_chosen = 'emilyalsentzer/Bio_ClinicalBERT'
    tokenizer = BertTokenizer.from_pretrained(bert_chosen)  
    #tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
    clinical_bert_token_size = 768
    #print(tokenizer)

    # Load pretrained model
    model = ModelOption(cfg.model.model_name,
                cfg.model.num_classes,
                freeze=cfg.model.freeze_weights,
                num_freezed_layers=cfg.model.num_frozen_layers,
                dropout=cfg.model.dropout,
                embedding_bool=cfg.model.embedding_bool,
                pool_algorithm=cfg.model.pool_algorithm
                )

    hidden_space_len = cfg.model.hidden_space_len

    net = MIL_model(model, hidden_space_len)

    net.load_state_dict(checkpoint_moco["encoder_state_dict"], strict=False)
    net.to(device)
    net.eval()

    for name, param in net.conv_layers.named_parameters():
        #if '10' in name or '11' in name: 
        param.requires_grad = False

    # Data augmentation
    prob = cfg.data_augmentation.prob

    pipeline_transform_local = A.Compose([
            A.VerticalFlip(p=prob),
            A.HorizontalFlip(p=prob),
            A.RandomRotate90(p=prob),
            #A.ElasticTransform(alpha=0.1,p=prob),
            A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-20,10),val_shift_limit=(-8,8),p=prob),
            ])

    # Data normalization
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev),
        transforms.Resize(size=(model.resize_param, model.resize_param),
        antialias=True)
    ])

    # Loss function
    criterion = getattr(torch.nn, cfg.training.criterion)()

    # Optimizer

    optimizer = getattr(torch.optim, cfg.training.optimizer)
    optimizer = optimizer(net.parameters(), **cfg.training.optimizer_args)

    # scheduler = getattr(torch.optim.lr_scheduler, cfg.training.lr_scheduler)
    # scheduler = scheduler(optimizer, **cfg.training.lr_scheduler_args)

    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight",'prelu']
    # emb_par = ["img_embeddings_encoder.weight", "embedding_fc.weight"]

    # param_optimizer_CNN = list(net.named_parameters())
    # print(len(param_optimizer_CNN))
    # #print(param_optimizer)
    # no_decay = ["prelu", "bias", "LayerNorm.bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters_CNN = [
    # 	{"params": [p for n, p in param_optimizer_CNN if not any(nd in n for nd in no_decay) and not any(nd in n for nd in emb_par)], "weight_decay": wt_decay},
    # 	{"params": [p for n, p in param_optimizer_CNN if any(nd in n for nd in no_decay) and not any(nd in n for nd in emb_par)], "weight_decay": 0.0,},
    # 	{"params": [p for n, p in param_optimizer_CNN if any(nd in n for nd in emb_par)], "weight_decay": wt_decay, 'lr': lr},
    # ]

    # #optimizer_CNN = optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wt_decay, amsgrad=True)
    # optimizer_CNN = optim.Adam(optimizer_grouped_parameters_CNN,lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wt_decay, amsgrad=True)
    #optimizer_CNN = AdamW(optimizer_grouped_parameters_CNN,lr = lr,eps=1e-8)

    # Training 
    epoch = 0
    early_stop = cfg.training.early_stop
    early_stop_cont = 0
    batch_size = cfg.dataloader.batch_size

    iterations = int(len(train_dataset))+1
    best_loss = 100000.0
    cont_iterations_tot = 0
    start_time = time.time()

    while (epoch<cfg.training.epochs and early_stop_cont<early_stop):

        # wandb.log({"epoch": epoch + 1})

        start_time_epoch = time.time()
        total_iters = 0
        train_loss = 0.0
        

        filenames_wsis = []
        pred_scc = []
        pred_nscc_ademo = []
        pred_nscc_squamoous = []
        pred_normal = []

        samples_fname = []
        positive_samples_fname = []
        negative_samples_fname = []
        hard_negative_samples_fname = []

        y_pred = []
        y_true = []

        dataloader_iterator = iter(training_generator_bag)

        net.train()

        # if (flag_dataset=='finetune_pretrain'):
        #     for param in net.embedding.parameters():
        #         param.requires_grad = False

        #     for param in net.attention.parameters():
        #         param.requires_grad = False
                
        #     for param in net.embedding_before_fc.parameters():
        #         param.requires_grad = False		

        total_params = sum(p.numel() for p in net.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} training parameters CNN.')

        for i in range(iterations):
            logging.info('[%d], %d / %d ' % (epoch + 1, i, iterations))
            try:
                wsi_id, labels = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(training_generator_bag)
                wsi_id, labels = next(dataloader_iterator)
                #inputs: bags, labels: labels of the bags
            
            wsi_id = wsi_id[0]
            labels = torch.stack(labels)
            labels_np = labels.cpu().numpy().flatten()

            labels_local = labels.float().flatten().to(device, non_blocking=True)

            # print("[" + str(i) + "/" + str(len(train_dataset)) + "], " + "inputs_bag: " + str(wsi_id))
            # print("labels: " + str(labels_np))
        

            #pipeline_transform = generate_transformer(labels_np[i_wsi])
            pipeline_transform = generate_transformer(labels_np)

            training_generator_instance = get_generator_instances(patches_path[wsi_id], 
                                                                  preprocess,
                                                                  batch_size, 
                                                                  pipeline_transform,
                                                                  cfg.dataloader.num_workers) 
            
            n_elems = len(patches_train[wsi_id])                                            
            net.eval()

            features = []
            with torch.no_grad():
                for instances in training_generator_instance:
                    instances = instances.to(device, non_blocking=True)

                    # forward + backward + optimize
                    feats = net.conv_layers(instances)
                    feats = feats.view(-1, net.fc_input_features)
                    feats_np = feats.cpu().data.numpy()

                    features = np.append(features,feats_np)

                    #del instances
                #del instances
            features_np = np.reshape(features,(n_elems, net.fc_input_features))

            #torch.cuda.empty_cache()
            #del features, feats

            net.train()
            net.zero_grad(set_to_none=True)

            inputs_embedding = torch.tensor(features_np, requires_grad=True).float().to(device, non_blocking=True)

            logits_img, cls_img = net(None, inputs_embedding)
            
            #loss img
            loss_img = criterion(logits_img, labels_local)
            
            #loss_classification = (loss_img + loss_neg) / 2
            loss = loss_img

            sigmoid_output_img = F.sigmoid(logits_img)
            outputs_wsi_np_img = sigmoid_output_img.cpu().data.numpy()
            train_loss = train_loss + ((1 / (i+1)) * (loss.item() - train_loss))
            
            total_iters = total_iters + 1
            cont_iterations_tot = cont_iterations_tot + 1

            # if (total_iters%200==True):
                # wandb.log({"iterations": cont_iterations_tot})
                # wandb.define_metric("train/loss_iter", step_metric="iterations")
                # wandb.log({"train/loss_iter": train_loss})

            loss.backward() 

            optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            #bert_model.zero_grad()

            logging.info("pred_img: " + str(outputs_wsi_np_img))
            logging.info("train_loss: " + str(train_loss))

            filenames_wsis = np.append(filenames_wsis, wsi_id)
            pred_scc = np.append(pred_scc, outputs_wsi_np_img[0])
            pred_nscc_ademo = np.append(pred_nscc_ademo, outputs_wsi_np_img[1])
            pred_nscc_squamoous = np.append(pred_nscc_squamoous, outputs_wsi_np_img[2])
            pred_normal = np.append(pred_normal, outputs_wsi_np_img[3])

            output_norm = np.where(outputs_wsi_np_img > 0.5, 1, 0)

            y_pred = np.append(y_pred, output_norm)
            y_true = np.append(y_true, labels_np)

            micro_accuracy_train = accuracy_micro(y_true, y_pred)
            logging.info("micro_accuracy " + str(micro_accuracy_train))    

        accuracy_train = accuracy_macro(y_true, y_pred)
        # wandb.define_metric("train/loss", step_metric="epoch")
        # wandb.log({"train/loss": train_loss})
        # wandb.define_metric("train/accuracy", step_metric="epoch")
        # wandb.log({"train/accuracy": micro_accuracy_train})

        #save_training predictions
        filename_training_predictions = Path(outputdir / f"training_predictions_{epoch}.csv")

        File = {'filenames': filenames_wsis,
                'pred_scc': pred_scc, 
                'pred_nscc_ademo': pred_nscc_ademo,
                'pred_nscc_squamoous': pred_nscc_squamoous, 
                'pred_normal': pred_normal}

        df_predictions = pd.dataframe.from_dict(File)
        df_predictions.to_csv(filename_training_predictions)

        #bert_model.eval()

        samples_fname = []
        positive_samples_fname = []
        negative_samples_fname = []
        hard_negative_samples_fname = []

        logging.info("== Validation ==")
        valid_loss, accuracy_validation = evaluate_validation_set(net, 
                                                                criterion, 
                                                                epoch, 
                                                                validation_generator_bag,
                                                                features_valid)
      
        # wandb.define_metric("validation/loss", step_metric="epoch")
        # wandb.log({"validation/loss": valid_loss})
        # wandb.define_metric("validation/accuracy", step_metric="epoch")
        # wandb.log({"validation/accuracy": accuracy_validation})

        # Create directories for the outputs
        outputdir_results = Path(outputdir /
                                 cfg.dataset.magnification / 
                                 cfg.model.model_name)
        Path(outputdir_results).mkdir(exist_ok=True, parents=True)

        model_filename = Path(outputdir_results / f"{cfg.experiment_name}.pt")
        model_filename_temp = Path(outputdir_results / f"{cfg.experiment_name}_temporary.pt")
        model_weights_filename_checkpoint = Path(outputdir_results /'checkpoint.pt')

        if (best_loss>valid_loss):
            early_stop_cont = 0
            logging.info ("=> Saving a new best model")
            logging.info("previous loss : " + str(best_loss) + ", new loss function: " + str(valid_loss))
            best_loss = valid_loss
            best_epoch = epoch

            try:
                torch.save({'epoch': best_epoch,
                            'encoder_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            # 'scheduler_state_dict': scheduler.state_dict(),
                            'train_loss': train_loss,
                            'valid_loss': best_loss},
                            model_filename,
                            _use_new_zipfile_serialization=False)
            except:
                torch.save({'epoch': best_epoch,
                            'encoder_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            # 'scheduler_state_dict': scheduler.state_dict(),
                            'train_loss': train_loss,
                            'valid_loss': best_loss},
                            model_filename)   
                    
            filename_training_predictions = Path(outputdir_results / 'training_predictions_best.csv')

            File = {'filenames': filenames_wsis,
                'pred_scc': pred_scc, 
                'pred_nscc_ademo': pred_nscc_ademo,
                'pred_nscc_squamoous': pred_nscc_squamoous, 
                'pred_normal': pred_normal}

            df_predictions = pd.dataframe.from_dict(File)
            df_predictions.to_csv(filename_training_predictions)

        else:
            early_stop_cont = early_stop_cont+1

        #save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'train_loss': train_loss,
            'valid_loss': best_loss},
            model_weights_filename_checkpoint)

        epoch = epoch+1
        if (early_stop_cont == early_stop):
            logging.info("======== EARLY STOPPING ========")

        message = timer(start_time_epoch, time.time())
        logging.info(f"Time to complete epoch {epoch + 1} is {message}" )

    message = timer(start_time, time.time())
    logging.info(f"Training complete in {message}" )
    logging.info(f"Best loss: {best_loss} at {best_epoch + 1}")

    torch.cuda.empty_cache()

    wandb.finish()


if __name__ == '__main__':
    main()
