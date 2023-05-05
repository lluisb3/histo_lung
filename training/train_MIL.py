from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import albumentations as A
import time
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.data
from torch.optim import Adam
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve, average_precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from natsort import natsorted
from transformers import BertModel, BertPreTrainedModel, RobertaConfig, RobertaModel
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
#from transformers.modeling_longformer import LongformerSelfAttention
from transformers import LongformerConfig, LongformerModel, LongformerSelfAttention
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification, AutoTokenizer

from ast import literal_eval
from database import Dataset_bag_MIL, Balanced_Multimodal
from training.mil import MIL_model
from training.models import ModelOption
from training.utils_trainig import yaml_load, initialize_wandb, edict2dict, get_generator_instances
import yaml
from utils import timer
import wandb
import logging
import click

thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# torch.backends.cudnn.benchmark = True


def focal_binary_cross_entropy(logits, targets, num_label, gamma=2):
    l = logits.reshape(-1)
    t = targets.reshape(-1)
    p = torch.sigmoid(l)
    p = torch.where(t >= 0.5, p, 1-p)
    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    loss = logp*((1-p)**gamma)
    loss = num_label*loss.mean()
    return loss


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

def generate_transformer(prob = 0.5):
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


# def validation_1_epoch(cfg,
#                        net,
#                        criterion,
#                        generator,
#                        patches_validation,
#                        preprocess,
#                        iterations,
#                        epoch):

def validation_1_epoch(cfg,
                       net,
                       criterion,
                       generator,
                       iterations,
                       epoch,
                       featuresdir):
    
    logging.info("== Validation ==")

    validation_loss = 0.0
    
    #accumulator for validation set
    filenames_wsis = []
    pred_scc = []
    pred_nscc_adeno = []
    pred_nscc_squamous = []
    pred_normal = []

    y_pred = []
    y_true = []
    scores_pred = []

    net.eval()
    
    dataloader_iterator = iter(generator)
    with torch.no_grad():
        for i in range(iterations):
            logging.info(f"[{epoch + 1}], {i + 1} / {iterations}")
            try:
                wsi_id, labels = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(generator)
                wsi_id, labels = next(dataloader_iterator)
                #inputs: bags, labels: labels of the bags

            wsi_id = wsi_id[0]
            labels = torch.stack(labels)
            labels_np = labels.cpu().numpy().flatten()

            labels_local = labels.float().flatten().to(device, non_blocking=True)

            features_np = np.load(datadir / "Saved_features" / featuresdir / f"{wsi_id}.npy")

            # validation_generator_instance = get_generator_instances(patches_validation[wsi_id], 
            #                                                         preprocess,
            #                                                         cfg.dataloader.batch_size, 
            #                                                         None,
            #                                                         cfg.dataloader.num_workers) 

            # n_elems = len(patches_validation[wsi_id])   

            # features = []
            
            # for instances in validation_generator_instance:
            #     instances = instances.to(device, non_blocking=True)

            #     # forward + backward + optimize
            #     feats = net.conv_layers(instances)
            #     feats = feats.view(-1, net.fc_input_features)
            #     feats_np = feats.cpu().data.numpy()

            #     features.extend(feats_np)

            # features_np = np.reshape(features,(n_elems, net.fc_input_features))

            inputs = torch.tensor(features_np).float().to(device, non_blocking=True)
        
            logits_img, _ = net(None, inputs)

            if cfg.training.criterion == "focal":
                loss_img = focal_binary_cross_entropy(logits_img,
                                                      labels_local,
                                                      cfg.model.num_classes)

            else:
                loss_img = criterion(logits_img, labels_local)

            sigmoid_output_img = F.sigmoid(logits_img)
            outputs_wsi_np_img = sigmoid_output_img.cpu().numpy()
            validation_loss = validation_loss + ((1 / (i+1)) * (loss_img.item() - validation_loss))
            
            logging.info(f"pred_img_logits: {outputs_wsi_np_img}")
            logging.info(f"validation_loss: {validation_loss}")

            filenames_wsis.append(wsi_id)
            pred_scc.append(outputs_wsi_np_img[0])
            pred_nscc_adeno.append(outputs_wsi_np_img[1])
            pred_nscc_squamous.append(outputs_wsi_np_img[2])
            pred_normal.append(outputs_wsi_np_img[3])

            output_norm = np.where(outputs_wsi_np_img > 0.5, 1, 0)
            logging.info(f"pred_img: {output_norm}")
            logging.info(f"y_true: {labels_np}")

            y_pred = np.append(y_pred, output_norm)
            y_true = np.append(y_true, labels_np)
            scores_pred= np.append(scores_pred, outputs_wsi_np_img)
            
            accuracy_valid = accuracy_score(y_true, y_pred)
            logging.info(f"Accuracy validation: {accuracy_valid}") 

        File = {'filenames': filenames_wsis,
        'pred_scc': pred_scc, 
        'pred_nscc_adeno': pred_nscc_adeno,
        'pred_nscc_squamous': pred_nscc_squamous, 
        'pred_normal': pred_normal}

        df_predictions = pd.DataFrame.from_dict(File)   
    
    return y_true, scores_pred, validation_loss, accuracy_valid, df_predictions

def train_1_epoch(cfg, 
                  net, 
                  criterion, 
                  optimizer,
                  scheduler, 
                  generator, 
                  patches_train, 
                  preprocess, 
                  iterations, 
                  epoch,
                  cont_iterations_tot,
                  data_augmentation):

    logging.info("== Training ==")

    total_iters = 0
    train_loss = 0.0
    
    filenames_wsis = []
    pred_scc = []
    pred_nscc_adeno = []
    pred_nscc_squamous = []
    pred_normal = []

    y_pred = []
    y_true = []
    scores_pred = []

    dataloader_iterator = iter(generator)

    net.train()

    # if (flag_dataset=='finetune_pretrain'):
    #     for param in net.embedding.parameters():
    #         param.requires_grad = False

    #     for param in net.attention.parameters():
    #         param.requires_grad = False
            
    #     for param in net.embedding_before_fc.parameters():
    #         param.requires_grad = False		

    for i in range(iterations):
        logging.info(f"[{epoch + 1}], {i + 1} / {iterations}")
        try:
            wsi_id, labels = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(generator)
            wsi_id, labels = next(dataloader_iterator)
            #inputs: bags, labels: labels of the bags
        
        wsi_id = wsi_id[0]
        labels = torch.stack(labels)
        labels_np = labels.cpu().numpy().flatten()

        labels_local = labels.float().flatten().to(device, non_blocking=True)

        # print("[" + str(i) + "/" + str(len(train_dataset)) + "], " + "inputs_bag: " + str(wsi_id))
        # print("labels: " + str(labels_np))
    

        #pipeline_transform = generate_transformer(labels_np[i_wsi])
        pipeline_transform = generate_transformer()

        if data_augmentation:
            
            training_generator_instance = get_generator_instances(patches_train[wsi_id], 
                                                                preprocess,
                                                                cfg.dataloader.batch_size, 
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
                    feats_np = feats.cpu().numpy()

                    features.extend(feats_np)

            features_np = np.reshape(features,(n_elems, net.fc_input_features))

        else:
            features_np = np.load(datadir / "Saved_features" / cfg.data_augmentation.featuresdir /
                                  f"{wsi_id}.npy")

        net.train()
        net.zero_grad(set_to_none=True)

        inputs_embedding = torch.tensor(features_np,
                                        requires_grad=True).float().to(device, non_blocking=True)

        logits_img, cls_img = net(None, inputs_embedding)
        
        if cfg.training.criterion == "focal":
            loss_img = focal_binary_cross_entropy(logits_img, labels_local, cfg.model.num_classes)

        else:
            loss_img = criterion(logits_img, labels_local)
        
        loss = loss_img

        sigmoid_output_img = F.sigmoid(logits_img)
        outputs_wsi_np_img = sigmoid_output_img.cpu().data.numpy()
        train_loss = train_loss + ((1 / (i+1)) * (loss.item() - train_loss))
        
        total_iters = total_iters + 1
        cont_iterations_tot = cont_iterations_tot + 1
        
        if (total_iters%200==True):
            if cfg.wandb.enable:
                wandb.log({"iterations": cont_iterations_tot})
                wandb.define_metric("train/loss_iter", step_metric="iterations")
                wandb.log({"train/loss_iter": train_loss})

        loss.backward() 

        optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        net.zero_grad(set_to_none=True)
        #bert_model.zero_grad()

        logging.info(f"pred_img_logits: {outputs_wsi_np_img}")
        logging.info(f"train_loss: {train_loss}")

        filenames_wsis.append(wsi_id)
        pred_scc.append(outputs_wsi_np_img[0])
        pred_nscc_adeno.append(outputs_wsi_np_img[1])
        pred_nscc_squamous.append(outputs_wsi_np_img[2])
        pred_normal.append(outputs_wsi_np_img[3])

        output_norm = np.where(outputs_wsi_np_img > 0.5, 1, 0)
        logging.info(f"pred_img: {output_norm}")
        logging.info(f"y_true: {labels_np}")

        y_pred = np.append(y_pred, output_norm)
        y_true = np.append(y_true, labels_np)
        
        scores_pred= np.append(scores_pred, outputs_wsi_np_img)

        micro_accuracy_train = accuracy_score(y_true, y_pred)
        logging.info(f"Accuracy train: {micro_accuracy_train}")

    scheduler.step()

    File = {'filenames': filenames_wsis,
            'pred_scc': pred_scc, 
            'pred_nscc_adeno': pred_nscc_adeno,
            'pred_nscc_squamous': pred_nscc_squamous, 
            'pred_normal': pred_normal}

    df_predictions = pd.DataFrame.from_dict(File)
        
    return y_true, scores_pred, train_loss, micro_accuracy_train, df_predictions


def train(cfg,
          net,
          criterion,
          optimizer,
          scheduler, 
          train_dataset, 
          validation_dataset, 
          training_generator_bag,
          patches_train,
          validation_generator_bag,
          patches_validation,
          preprocess,
          outputdir):

    # Start Training 
    logging.info(f"== Start training {cfg.experiment_name} ==")
    epoch = 0
    early_stop = cfg.training.early_stop
    early_stop_cont = 0

    iterations_train = int(len(train_dataset))
    iterations_valid = int(len(validation_dataset))

    best_loss = 100000.0
    cont_iterations_tot = 0
    start_time = time.time()

    if cfg.training.resume_training:
        chkptdir = Path(thispath.parent.parent / 
                "trained_models" / 
                "MIL" / 
                cfg.experiment_name)
        
        checkpoint = torch.load(chkptdir)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
    else:
        epoch = 0 
        best_loss = 100000.0

    while (epoch<cfg.training.epochs and early_stop_cont<early_stop):
        if cfg.wandb.enable:
            wandb.log({"epoch": epoch + 1})
            wandb.define_metric("train/lr", step_metric="epoch")
            wandb.log({"train/lr": optimizer.param_groups[0]["lr"]})

        start_time_epoch = time.time()

        # Training
        
        y_true_tr, scores_tr, train_loss, accuracy_train, df_train = train_1_epoch(
                                                                      cfg,
                                                                      net,
                                                                      criterion,
                                                                      optimizer,
                                                                      scheduler,
                                                                      training_generator_bag,
                                                                      patches_train,
                                                                      preprocess,
                                                                      iterations_train,
                                                                      epoch,
                                                                      cont_iterations_tot,
                                                                      cfg.data_augmentation.boolean)
        #save_training predictions
        filename_training_predictions = Path(outputdir / f"training_predictions_{epoch + 1}.csv")
        df_train.to_csv(filename_training_predictions)
        
        arange_like_train = np.arange(len(y_true_tr))
        names = ["SCC", "NSCC Adeno", "NSCC Squamous", "No Cancer"]

        y_pred_tr = np.where(scores_tr > 0.5, 1, 0)

        y_pred_tr_reshape = y_pred_tr.reshape(len(train_dataset), 4)
        y_true_tr_reshape = y_true_tr.reshape(len(train_dataset), 4)

        # Compute ROC curve and ROC area for each class
        fpr_train = {}
        tpr_train = {}
        roc_auc_train = {}
        precision_train = {}
        recall_train = {}
        avg_precision_train = {}
        for i in range(4):
            fpr_train[i], tpr_train[i], _ = roc_curve(
                                          y_true_tr[arange_like_train%4 == 0 + i],
                                          scores_tr[arange_like_train%4 == 0 + i])
            roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])

            precision_train[i], recall_train[i], _ = precision_recall_curve(
                                                y_true_tr[arange_like_train%4 == 0 + i],
                                                scores_tr[arange_like_train%4 == 0 + i])
            avg_precision_train[i] = average_precision_score(
                                                y_true_tr[arange_like_train%4 == 0 + i],
                                                scores_tr[arange_like_train%4 == 0 + i])


        # Compute micro-average ROC curve and ROC area
        fpr_train["micro"], tpr_train["micro"], _ = roc_curve(y_true_tr, scores_tr)
        roc_auc_train["micro"] = auc(fpr_train["micro"], tpr_train["micro"])
        
        precision_train["micro"], recall_train["micro"], _ = precision_recall_curve(y_true_tr,
                                                                        scores_tr)
        avg_precision_train["micro"] = average_precision_score(y_true_tr_reshape,
                                                             y_pred_tr_reshape,
                                                             average="micro")

        f1_micro_train = f1_score(y_pred_tr_reshape, y_true_tr_reshape, average="micro")
        f1_macro_train = f1_score(y_pred_tr_reshape, y_true_tr_reshape, average="macro")
        f1_weighted_train = f1_score(y_pred_tr_reshape, y_true_tr_reshape, average="weighted")

        # Plot ROC curve
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="whitegrid", rc=custom_params)
        plt.figure()
        plt.plot(fpr_train["micro"], tpr_train["micro"],
                label='micro-average ROC curve (AUC = {0:0.2f})'
                    ''.format(roc_auc_train["micro"]))
        for i in range(4):
                plt.plot(fpr_train[i],
                    tpr_train[i],
                    label=f"ROC curve {names[i]} (AUC = {roc_auc_train[i]:0.2f})")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (Fpr)")
        plt.ylabel("True Positive Rate (Tpr)")
        plt.title("Train ROC Lung Cancer Multiclass")
        plt.legend(loc="lower right")
        plt.savefig(outputdir / f"train_{epoch + 1}_roc.svg")
        plt.close()

        # Plot PR curve
        sns.set_theme(style="white", rc=custom_params)
        plt.figure()
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
                plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        plt.plot(recall_train["micro"], precision_train["micro"],
                label='micro-average PR curve (AP = {0:0.2f})'
                    ''.format(avg_precision_train["micro"]))
        for i in range(4):
                plt.plot(recall_train[i],
                    precision_train[i],
                    label=f"PR curve {names[i]} (AP = {avg_precision_train[i]:0.2f})")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Train PR curve Lung Cancer Multiclass")
        plt.legend(loc="lower left")
        plt.savefig(outputdir / f"train_{epoch + 1}_pr_curve.svg")
        plt.close()

        if cfg.wandb.enable:
            wandb.define_metric("train/loss", step_metric="epoch")
            wandb.log({"train/loss": train_loss})
            wandb.define_metric("train/accuracy", step_metric="epoch")
            wandb.log({"train/accuracy": accuracy_train})
            wandb.define_metric("train/auc_micro", step_metric="epoch")
            wandb.log({"train/auc_micro": roc_auc_train["micro"]})
            wandb.define_metric("train/f1_score_micro", step_metric="epoch")
            wandb.log({"train/f1_score_micro": f1_micro_train})
            wandb.define_metric("train/f1_score_macro", step_metric="epoch")
            wandb.log({"train/f1_score_macro": f1_macro_train})
            wandb.define_metric("train/f1_score_weighted", step_metric="epoch")
            wandb.log({"train/f1_score_weighted": f1_weighted_train})
            
        logging.info(f"Accuracy train Epoch {epoch + 1}: {accuracy_train}")

        # Validation
        y_true_vd, scores_vd, valid_loss, accuracy_valid, df_valid = validation_1_epoch(
                                                                  cfg,
                                                                  net,
                                                                  criterion,
                                                                  validation_generator_bag,
                                                                  iterations_valid,
                                                                  epoch,
                                                                  cfg.data_augmentation.featuresdir)
        
        # Save validation predictions
        filename_validation_predictions = Path(outputdir / f"validation_predictions_{epoch + 1}.csv")
        df_valid.to_csv(filename_validation_predictions) 

        arange_like_valid = np.arange(len(y_true_vd))

        y_pred_vd = np.where(scores_vd > 0.5, 1, 0)

        y_pred_vd_reshape = y_pred_vd.reshape(len(validation_dataset), 4)
        y_true_vd_reshape = y_true_vd.reshape(len(validation_dataset), 4)

        # Compute ROC curve and ROC area for each class
        fpr_valid = {}
        tpr_valid = {}
        roc_auc_valid = {}
        precision_valid = {}
        recall_valid = {}
        avg_precision_valid = {}
        for i in range(4):
                fpr_valid[i], tpr_valid[i], _ = roc_curve(
                                            y_true_vd[arange_like_valid%4 == 0 + i],
                                            scores_vd[arange_like_valid%4 == 0 + i])
                roc_auc_valid[i] = auc(fpr_valid[i], tpr_valid[i])

                precision_valid[i], recall_valid[i], _ = precision_recall_curve(
                                                    y_true_vd[arange_like_valid%4 == 0 + i],
                                                    scores_vd[arange_like_valid%4 == 0 + i])
                avg_precision_valid[i] = average_precision_score(
                                                    y_true_vd[arange_like_valid%4 == 0 + i],
                                                    scores_vd[arange_like_valid%4 == 0 + i])

        # Compute micro-average ROC curve and ROC area
        fpr_valid["micro"], tpr_valid["micro"], _ = roc_curve(y_true_vd, scores_vd)
        roc_auc_valid["micro"] = auc(fpr_valid["micro"], tpr_valid["micro"])
        
        precision_valid["micro"], recall_valid["micro"], _ = precision_recall_curve(y_true_vd,
                                                                        scores_vd)
        avg_precision_valid["micro"] = average_precision_score(y_true_vd_reshape,
                                                             y_pred_vd_reshape,
                                                             average="micro")

        f1_micro_valid = f1_score(y_pred_vd_reshape, y_true_vd_reshape, average="micro")
        f1_macro_valid = f1_score(y_pred_vd_reshape, y_true_vd_reshape, average="macro")
        f1_weighted_valid = f1_score(y_pred_vd_reshape, y_true_vd_reshape, average="weighted")

        # Plot ROC curve
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="whitegrid", rc=custom_params)
        plt.figure()
        plt.plot(fpr_valid["micro"], tpr_valid["micro"],
                label='micro-average ROC curve (AUC = {0:0.2f})'
                    ''.format(roc_auc_valid["micro"]))
        for i in range(4):
                plt.plot(fpr_valid[i],
                    tpr_valid[i],
                    label=f"ROC curve {names[i]} (AUC = {roc_auc_valid[i]:0.2f})")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (Fpr)")
        plt.ylabel("True Positive Rate (Tpr)")
        plt.title("Validation ROC Lung Cancer Multiclass")
        plt.legend(loc="lower right")
        plt.savefig(outputdir / f"valid_{epoch + 1}_roc.svg")
        plt.close()

        # Plot PR curve
        sns.set_theme(style="white", rc=custom_params)
        plt.figure()
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
                plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        plt.plot(recall_valid["micro"], precision_valid["micro"],
                label='micro-average PR curve (AP = {0:0.2f})'
                    ''.format(avg_precision_valid["micro"]))
        for i in range(4):
                plt.plot(recall_valid[i],
                    precision_valid[i],
                    label=f"PR curve {names[i]} (AP = {avg_precision_valid[i]:0.2f})")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Validation PR curve Lung Cancer Multiclass")
        plt.legend(loc="lower left")
        plt.savefig(outputdir / f"valid_{epoch + 1}_pr_curve.svg")
        plt.close()
    
        # Wandb
        if cfg.wandb.enable: 
            wandb.define_metric("validation/loss", step_metric="epoch")
            wandb.log({"validation/loss": valid_loss})
            wandb.define_metric("validation/accuracy", step_metric="epoch")
            wandb.log({"validation/accuracy": accuracy_valid})

            wandb.define_metric("validation/auc_micro", step_metric="epoch")
            wandb.log({"validation/auc_micro": roc_auc_valid["micro"]})
            wandb.define_metric("validation/f1_score_micro", step_metric="epoch")
            wandb.log({"validation/f1_score_micro": f1_micro_valid})
            wandb.define_metric("validation/f1_score_macro", step_metric="epoch")
            wandb.log({"validation/f1_score_macro": f1_macro_valid})
            wandb.define_metric("validation/f1_score_weighted", step_metric="epoch")
            wandb.log({"validation/f1_score_weighted": f1_weighted_valid})

        # Create directories for the outputs
        outputdir_results = Path(outputdir /
                                 cfg.dataset.magnification / 
                                 cfg.model.model_name)
        Path(outputdir_results).mkdir(exist_ok=True, parents=True)

        model_filename = Path(outputdir_results / f"{cfg.experiment_name}.pt")

        if (best_loss > valid_loss):
            early_stop_cont = 0
            logging.info ("== Saving a new best model ==")
            logging.info(f"Previous loss: {str(best_loss)}, New loss function: {str(valid_loss)}")
            best_loss = valid_loss
            best_epoch = epoch

            try:
                torch.save({'epoch': best_epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'train_loss': train_loss,
                            'valid_loss': best_loss},
                            model_filename,
                            _use_new_zipfile_serialization=False)
            except:
                torch.save({'epoch': best_epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'train_loss': train_loss,
                            'valid_loss': best_loss},
                            model_filename)   
            
            # Save best predictions
            best_training_predictions = Path(outputdir_results / 
                                             "training_predictions_best.csv")
            df_train.to_csv(best_training_predictions)

            best_validation_predictions = Path(outputdir_results / 
                                               "validation_predictions_best.csv")
            df_valid.to_csv(best_validation_predictions) 

        else:
            early_stop_cont = early_stop_cont+1

        model_weights_filename_checkpoint = Path(outputdir /'checkpoint.pt')
        #save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': valid_loss,
            'train_accuracy': accuracy_train,
            'valid_accuracy': accuracy_valid},
            model_weights_filename_checkpoint)

        epoch = epoch + 1
        if (early_stop_cont == early_stop):
            logging.info("======== EARLY STOPPING ========")

        message = timer(start_time_epoch, time.time())
        logging.info(f"Time to complete epoch {epoch + 1} is {message}" )

    message = timer(start_time, time.time())
    logging.info(f"Training complete in {message}" )
    logging.info(f"Best loss: {best_loss} at {best_epoch + 1}")

    torch.cuda.empty_cache()


@click.command()
@click.option(
    "--config_file",
    default="config_MIL",
    prompt="Name of the config file without extension",
    help="Name of the config file without extension",
)
@click.option(
    "--exp_name_moco",
    default="MoCo_resnet34_scheduler_51015",
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

    # Save config parameters for experiment
    with open(Path(f"{outputdir}/config_{cfg.experiment_name}.yml"), 'w') as yaml_file:
        yaml.dump(edict2dict(cfg), yaml_file, default_flow_style=False)

    # wandb login
    if cfg.wandb.enable:
            wandb.login()
            # key = os.environ.get("WANDB_API_KEY")
            wandb_run = initialize_wandb(cfg, outputdir)
            wandb_run.define_metric("epoch", summary="max")

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

    # Loading Data Split
    k = 5

    data_split = pd.read_csv(Path(datadir / f"{k}_fold_crossvalidation_data_split.csv"), index_col=0)
    train_dataset_k = []
    validation_dataset_k = []
    train_labels_k = []
    validation_labels_k = []

    for fold, _ in data_split.iterrows():
        train_wsi = literal_eval(data_split.loc[fold]["images_train"])
        validation_wsi = literal_eval(data_split.loc[fold]["images_validation"])
        labels_train = literal_eval(data_split.loc[fold]["labels_train"])
        labels_validation = literal_eval(data_split.loc[fold]["labels_validation"])
        train_dataset_k.append(train_wsi)
        validation_dataset_k.append(validation_wsi)
        train_labels_k.append(labels_train)
        validation_labels_k.append(labels_validation)

    # Load fold 0
    train_dataset = train_dataset_k[0]
    validation_dataset = validation_dataset_k[0]
    train_labels = train_labels_k[0]
    validation_labels = validation_labels_k[0]
    
    # Discard WSI with less than 10 patches
    pyhistdir = Path(datadir / "Mask_PyHIST_v2") 

    metadata_dataset = pd.read_csv(pyhistdir / "metadata_slides.csv", index_col=0,
                                   dtype={"ID wsi": str})

    discard_wsi_dataset = []
    if (metadata_dataset['number_filtered_patches'] < 10).any():
        for index, row in metadata_dataset.iterrows():
            if row['number_filtered_patches'] < 10:
                discard_wsi_dataset.append(index)
        logging.info(f"There is {len(discard_wsi_dataset)} WSI discarded in train/valid becasue <10 patches")
        logging.info(discard_wsi_dataset)

    i = 0
    for image in train_dataset:
        if image in discard_wsi_dataset:
            train_dataset.pop(i)
            train_labels.pop(i)
        i+=1

    i = 0
    for image in validation_dataset:
        if image in discard_wsi_dataset:
            validation_dataset.pop(i)
            validation_labels.pop(i)
        i+=1

    # Load patches path
    dataset_path = natsorted([i for i in pyhistdir.rglob("*_densely_filtered_paths.csv") 
                              if "LungAOEC" in str(i)])

    patches_path = {}
    for wsi_patches_path in tqdm(dataset_path, desc="Selecting patches: "):

        csv_patch_path = pd.read_csv(wsi_patches_path).to_numpy()

        name = wsi_patches_path.parent.stem
        patches_path[name] = csv_patch_path
        # patches_names[name] = [filenames]

        # patches_names[name] = []
        # for instance in csv_patch_path:
        #         patches_names[name].append(str(instance).split("/")[-1])

    logging.info(f"Total number of WSI for train/validation {len(patches_path)}")

    patches_train = {}
    patches_validation = {}
    for value, key in zip(patches_path.values(), patches_path.keys()):
        if key in train_dataset:
            patches_train[key] = value
        if key in validation_dataset:
            patches_validation[key] = value

    logging.info(f"Total number of WSI for train {len(patches_train.values())}")
    logging.info(f"Total number of WSI for validation {len(patches_validation.values())}")

    # Load datasets
    batch_size_bag = cfg.dataloader.batch_size_bag

    sampler = Balanced_Multimodal
    train_data_labels = np.zeros([len(train_dataset), 5], dtype=float)
    train_data_labels[:, 0] = train_dataset
    train_data_labels[:, 1:] = np.array(train_labels)

    # params_train_bag = {'batch_size': batch_size_bag,
    #     'sampler': sampler(train_data_labels, alpha=0.25)}

    params_train_bag = {'batch_size': batch_size_bag,
                        'shuffle': True}

    params_valid_bag = {'batch_size': batch_size_bag, # len(validation_dataset),
                        'shuffle': False}

    training_set_bag = Dataset_bag_MIL(train_dataset, train_labels)
    training_generator_bag = DataLoader(training_set_bag, **params_train_bag)

    validation_set_bag = Dataset_bag_MIL(validation_dataset, validation_labels)
    validation_generator_bag = DataLoader(validation_set_bag, **params_valid_bag)

    # Load features from MoCo model
    experiment_name = exp_name_moco

    logging.info(f"== Loading MoCo from {experiment_name} ==")

    mocodir = Path(thispath.parent.parent / 
                "trained_models" / 
                "MoCo" / 
                experiment_name)

    cfg_moco = yaml_load(mocodir / f"config_{experiment_name}.yml")

    checkpoint_moco = torch.load(Path(mocodir /
                                cfg_moco.dataset.magnification / 
                                cfg_moco.model.model_name / 
                                f"{exp_name_moco}.pt"))


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

    total_params = sum(p.numel() for p in net.parameters())
    logging.info(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(f'{total_trainable_params:,} training parameters CNN.')

    # Data normalization
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev),
        transforms.Resize(size=(model.resize_param, model.resize_param),
        antialias=True)
    ])

    # Loss function
    if cfg.training.criterion == "focal":
        criterion = None
        logging.info("== Criterion: Focal Loss ==")

    else:
        if cfg.training.criterion_args.get('weights') is not None:
            weights = torch.tensor(cfg.training.criterion_args.weights,
                                dtype=torch.float,
                                device=device)
            criterion = getattr(torch.nn, cfg.training.criterion)(weight=weights)
            logging.info(f"== Criterion: {cfg.training.criterion} with weights ==")

        else:
            criterion = getattr(torch.nn, cfg.training.criterion)()
            logging.info(f"== Criterion: {cfg.training.criterion} ==")

    # Optimizer
    param_optimizer_cfg = cfg.training.optimizer_args
    
    lr = param_optimizer_cfg.lr
    wt_decay = param_optimizer_cfg.weight_decay
    betas = param_optimizer_cfg.betas
    eps = param_optimizer_cfg.eps
    amsgrad = param_optimizer_cfg.amsgrad

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight",'prelu']
    emb_par = ["img_embeddings_encoder.weight", "embedding_fc.weight"]

    param_optimizer_CNN = list(net.named_parameters())
    no_decay = ["prelu", "bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters_CNN = [
    	{"params": [p for n, p in param_optimizer_CNN if not any(nd in n for nd in no_decay) and not any(nd in n for nd in emb_par)], "weight_decay": wt_decay},
    	{"params": [p for n, p in param_optimizer_CNN if any(nd in n for nd in no_decay) and not any(nd in n for nd in emb_par)], "weight_decay": 0.0,},
    	{"params": [p for n, p in param_optimizer_CNN if any(nd in n for nd in emb_par)], "weight_decay": wt_decay, 'lr': lr},
    ]

    #optimizer_CNN = optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wt_decay, amsgrad=True)
    optimizer = Adam(optimizer_grouped_parameters_CNN, 
                     lr=lr, 
                     betas=betas, 
                     eps=eps, 
                     weight_decay=wt_decay, 
                     amsgrad=amsgrad)
    # optimizer_CNN = AdamW(optimizer_grouped_parameters_CNN,lr = lr,eps=1e-8)

    scheduler = getattr(torch.optim.lr_scheduler, cfg.training.lr_scheduler)
    scheduler = scheduler(optimizer, **cfg.training.lr_scheduler_args)

    # Training
    train(cfg,
          net,
          criterion,
          optimizer,
          scheduler, 
          train_dataset, 
          validation_dataset, 
          training_generator_bag,
          patches_train,
          validation_generator_bag,
          patches_validation,
          preprocess,
          outputdir)

    if cfg.wandb.enable:
        wandb.finish()


if __name__ == '__main__':
    main()
