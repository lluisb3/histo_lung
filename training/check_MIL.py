from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from ast import literal_eval
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from database import Dataset_bag_MIL
from training.mil import MIL_model
from training.models import ModelOption
from training.utils_trainig import yaml_load, get_generator_instances
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from natsort import natsorted
import matplotlib.pyplot as plt
import seaborn as sns

thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Load PyTorch model
experiment_name = "MIL_resnet101_0001_256"

dataset = "test"

modeldir = Path(thispath.parent.parent / "trained_models" / "MIL" / experiment_name)

cfg = yaml_load(modeldir / f"config_{experiment_name}.yml")

bestdir = Path(modeldir / cfg.dataset.magnification / cfg.model.model_name)

checkpoint = torch.load(bestdir / f"{experiment_name}.pt")

train_loss = checkpoint["train_loss"]
valid_loss = checkpoint["valid_loss"]
epoch = checkpoint["epoch"] + 1

print(f"Loaded encoder using as backbone {cfg.model.model_name} with a best loss in train "
      f"of {train_loss} and in validation of {valid_loss} at Epoch {epoch}")

labels = pd.read_csv(datadir / "labels.csv", index_col=0)

train_predictions = pd.read_csv(bestdir / "training_predictions_best.csv", index_col=0)
valid_predictions = pd.read_csv(bestdir / "validation_predictions_best.csv", index_col=0)


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

net.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
net.to(device)
net.eval()

# Loading Data Split
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


# Load Test labels
test_csv = pd.read_csv(Path(datadir / f"labels_test.csv"), index_col=0)
test_dataset = test_csv.index
test_dataset = [i.replace("/", "-") for i in test_dataset]
test_labels = test_csv.values

# Load Test patches
pyhistdir = Path(datadir / "Mask_PyHIST_v2")

dataset_path = natsorted([i for i in pyhistdir.rglob("*_densely_filtered_paths.csv")])
dataset_name = natsorted([i for i in pyhistdir.rglob("*_densely_filtered_metadata.csv")])

patches_path = {}
patches_names = {}
for wsi_patches_path, wsi_patches_names in tqdm(zip(dataset_path, dataset_name),
                                                desc="Selecting patches: "):

      csv_patch_path = pd.read_csv(wsi_patches_path).to_numpy()
      csv_patch_names = pd.read_csv(wsi_patches_names, index_col=0)

      names = csv_patch_names.index.to_numpy()
      
      name = wsi_patches_path.parent.stem
      patches_path[name] = csv_patch_path
      # patches_names[name] = [names]

      # patches_names[name] = []
      # for instance in csv_patch_path:
      #         patches_names[name].append(str(instance).split("/")[-1])

print(f"Total number of patches for train/validation/test {len(patches_path)}")

patches_train = {}
patches_validation = {}
patches_test = {}
for value, key in zip(patches_path.values(), patches_path.keys()):
      if key in train_dataset:
            patches_train[key] = value
      if key in validation_dataset:
            patches_validation[key] = value
      if key in test_dataset:
            patches_test[key] = value

# print(f"Total number of wsi for train {len(patches_train.values())}")
# print(f"Total number of wsi for validation {len(patches_validation.values())}")
print(f"Total number of patches for test {len(patches_test)}")

# Load datasets
batch_size_bag = cfg.dataloader.batch_size_bag

params_train_bag = {'batch_size': batch_size_bag,
                  'shuffle': False}

train_set_bag = Dataset_bag_MIL(train_dataset, train_labels)
train_generator_bag = DataLoader(train_set_bag, **params_train_bag)

params_valid_bag = {'batch_size': batch_size_bag,
                  'shuffle': False}

valid_set_bag = Dataset_bag_MIL(validation_dataset, validation_labels)
valid_generator_bag = DataLoader(valid_set_bag, **params_valid_bag)

params_test_bag = {'batch_size': batch_size_bag,
                  'shuffle': False}

test_set_bag = Dataset_bag_MIL(test_dataset, test_labels)
test_generator_bag = DataLoader(test_set_bag, **params_test_bag)

# Data normalization
preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev),
      transforms.Resize(size=(model.resize_param, model.resize_param),
      antialias=True)
])

if dataset == "test":
      # Test
      outputdir = Path(bestdir / "test")
      Path(outputdir).mkdir(exist_ok=True, parents=True)

      filenames_wsis = []
      pred_scc = []
      pred_nscc_adeno = []
      pred_nscc_squamous = []
      pred_normal = []

      y_pred = []
      y_true = []
      scores_pred = []

      iterations_test= int(len(test_dataset))
      dataloader_iterator = iter(test_generator_bag)
      with torch.no_grad():
            for i in range(iterations_test):
                  print(f"Iterations: {i + 1} / {iterations_test}")
                  try:
                        wsi_id, labels = next(dataloader_iterator)
                  except StopIteration:
                        dataloader_iterator = iter(test_generator_bag)
                        wsi_id, labels = next(dataloader_iterator)
                        #inputs: bags, labels: labels of the bags

                  wsi_id = wsi_id[0]
                  
                  labels_np = labels.cpu().numpy().flatten()

                  labels_local = labels.float().flatten().to(device, non_blocking=True)

                  test_generator_instance = get_generator_instances(patches_test[wsi_id], 
                                                                  preprocess,
                                                                  cfg.dataloader.batch_size, 
                                                                  None,
                                                                  cfg.dataloader.num_workers) 

                  n_elems = len(patches_test[wsi_id])   

                  features = []
                  
                  for instances in test_generator_instance:
                        instances = instances.to(device, non_blocking=True)

                        # forward + backward + optimize
                        feats = net.conv_layers(instances)
                        feats = feats.view(-1, net.fc_input_features)
                        feats_np = feats.cpu().data.numpy()

                        features.extend(feats_np)

                              #del instances
                        #del instances
                  features_np = np.reshape(features,(n_elems, net.fc_input_features))

                  inputs = torch.tensor(features_np).float().to(device, non_blocking=True)
                  
                  logits_img, _ = net(None, inputs)


                  sigmoid_output_img = F.sigmoid(logits_img)
                  outputs_wsi_np_img = sigmoid_output_img.cpu().numpy()

                  print(f"pred_img: {outputs_wsi_np_img}")

                  filenames_wsis.append(wsi_id)
                  pred_scc.append(outputs_wsi_np_img[0])
                  pred_nscc_adeno.append(outputs_wsi_np_img[1])
                  pred_nscc_squamous.append(outputs_wsi_np_img[2])
                  pred_normal.append(outputs_wsi_np_img[3])

                  output_norm = np.where(outputs_wsi_np_img > 0.5, 1, 0)

                  y_pred = np.append(y_pred, output_norm)
                  y_true = np.append(y_true, labels_np)
                  scores_pred= np.append(scores_pred, outputs_wsi_np_img)
                  
                  micro_accuracy_test = accuracy_score(y_true, y_pred)
                  print(f"micro_accuracy test {micro_accuracy_test}") 

            File = {'filenames': filenames_wsis,
            'pred_scc': pred_scc, 
            'pred_nscc_adeno': pred_nscc_adeno,
            'pred_nscc_squamous': pred_nscc_squamous, 
            'pred_normal': pred_normal}

            df_predictions = pd.DataFrame.from_dict(File)
            
            filename_test_predictions = Path(outputdir / f"test_predictions_{epoch + 1}.csv")
            df_predictions.to_csv(filename_test_predictions) 

            arange_like_predictions = np.arange(len(y_true))
            names = ["SCC", "NSCC_Adeno", "NSCC_Squamous", "Normal"]

            y_pred = np.where(scores_pred > 0.5, 1, 0)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            precision = {}
            recall = {}
            avg_precision = {}
            roc_auc = dict()
            for i in range(4):
                  fpr[i], tpr[i], _ = roc_curve(y_true[arange_like_predictions%4 == 0 + i],
                                                scores_pred[arange_like_predictions%4 == 0 + i])
                  roc_auc[i] = auc(fpr[i], tpr[i])

                  precision[i], recall[i], _ = precision_recall_curve(y_true[arange_like_predictions%4 == 0 + i],
                                                                      scores_pred[arange_like_predictions%4 == 0 + i])
                  avg_precision[i] = average_precision_score(y_true[arange_like_predictions%4 == 0 + i],
                                                             scores_pred[arange_like_predictions%4 == 0 + i])
                  print(scores_pred[arange_like_predictions%4 == 0 + i])
                  print(y_pred[arange_like_predictions%4 == 0 + i])
                  
                  accuracy = accuracy_score(y_true[arange_like_predictions%4 == 0 + i],
                                          y_pred[arange_like_predictions%4 == 0 + i])

                  print("== Final Metrics ==")
                  print(f"Accuracy {names[i]} = {accuracy:0.2f}")

                  bma = balanced_accuracy_score(y_true[arange_like_predictions%4 == 0 + i],
                                                y_pred[arange_like_predictions%4 == 0 + i])
                  kappa = cohen_kappa_score(y_true[arange_like_predictions%4 == 0 + i],
                                          y_pred[arange_like_predictions%4 == 0 + i])
                  print(f"BMA {names[i]} = {bma:0.2f}")
                  print(f"Kappa {names[i]} = {kappa:0.2f}")

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true, scores_pred)
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Plot ROC curve
            custom_params = {"axes.spines.right": False, "axes.spines.top": False}
            sns.set_theme(style="whitegrid", rc=custom_params)
            plt.figure()
            plt.plot(fpr["micro"], tpr["micro"],
                  label='micro-average ROC curve (area = {0:0.2f})'
                        ''.format(roc_auc["micro"]))
            for i in range(4):
                  plt.plot(fpr[i], tpr[i], label=f"ROC curve {names[i]} (AUC = {roc_auc[i]:0.2f})")

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate (Fpr)")
            plt.ylabel("True Positive Rate (Tpr)")
            plt.title("Test ROC Lung Cancer Multiclass")
            plt.legend(loc="lower right")
            plt.savefig(outputdir / f"test_{epoch + 1}_roc.svg")
            plt.clf()

            sns.set_theme(style="whitegrid", rc=custom_params)
            plt.figure()
            for i in range(4):
                  plt.plot(recall[i], precision[i], label=f"PR curve {names[i]} (AP = {avg_precision[i]:0.2f})")

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Test PR Lung Cancer Multiclass")
            plt.legend(loc="top right")
            plt.savefig(outputdir / f"test_{epoch + 1}_pr_curve.svg")
            plt.clf()

elif dataset == "valid":
      # Validation
      outputdir = Path(bestdir / "valid")
      Path(outputdir).mkdir(exist_ok=True, parents=True)

      filenames_wsis = []
      pred_scc = []
      pred_nscc_adeno = []
      pred_nscc_squamous = []
      pred_normal = []

      y_pred = []
      y_true = []
      scores_pred = []

      iterations_valid= int(len(validation_dataset))
      dataloader_iterator = iter(valid_generator_bag)
      with torch.no_grad():
            for i in range(iterations_valid):
                  print(f"Iterations: {i + 1} / {iterations_valid}")
                  try:
                        wsi_id, labels = next(dataloader_iterator)
                  except StopIteration:
                        dataloader_iterator = iter(valid_generator_bag)
                        wsi_id, labels = next(dataloader_iterator)
                        #inputs: bags, labels: labels of the bags

                  wsi_id = wsi_id[0]
                  labels = torch.stack(labels)
                  labels_np = labels.cpu().numpy().flatten()

                  labels_local = labels.float().flatten().to(device, non_blocking=True)

                  valid_generator_instance = get_generator_instances(patches_validation[wsi_id], 
                                                                     preprocess,
                                                                     cfg.dataloader.batch_size, 
                                                                     None,
                                                                     cfg.dataloader.num_workers) 

                  n_elems = len(patches_validation[wsi_id])   

                  features = []
                  
                  for instances in valid_generator_instance:
                        instances = instances.to(device, non_blocking=True)

                        # forward + backward + optimize
                        feats = net.conv_layers(instances)
                        feats = feats.view(-1, net.fc_input_features)
                        feats_np = feats.cpu().data.numpy()

                        features.extend(feats_np)

                              #del instances
                        #del instances
                  features_np = np.reshape(features,(n_elems, net.fc_input_features))

                  inputs = torch.tensor(features_np).float().to(device, non_blocking=True)
                  
                  logits_img, _ = net(None, inputs)


                  sigmoid_output_img = F.sigmoid(logits_img)
                  outputs_wsi_np_img = sigmoid_output_img.cpu().numpy()

                  print(f"pred_img: {outputs_wsi_np_img}")

                  filenames_wsis.append(wsi_id)
                  pred_scc.append(outputs_wsi_np_img[0])
                  pred_nscc_adeno.append(outputs_wsi_np_img[1])
                  pred_nscc_squamous.append(outputs_wsi_np_img[2])
                  pred_normal.append(outputs_wsi_np_img[3])

                  output_norm = np.where(outputs_wsi_np_img > 0.5, 1, 0)

                  y_pred = np.append(y_pred, output_norm)
                  y_true = np.append(y_true, labels_np)
                  scores_pred= np.append(scores_pred, outputs_wsi_np_img)
                  
                  accuracy_valid = accuracy_score(y_true, y_pred)
                  print(f"micro_accuracy validation {accuracy_valid}") 

            File = {'filenames': filenames_wsis,
            'pred_scc': pred_scc, 
            'pred_nscc_adeno': pred_nscc_adeno,
            'pred_nscc_squamous': pred_nscc_squamous, 
            'pred_normal': pred_normal}

            df_predictions = pd.DataFrame.from_dict(File)
            
            filename_valid_predictions = Path(outputdir / f"valid_predictions_{epoch + 1}.csv")
            df_predictions.to_csv(filename_valid_predictions) 

            arange_like_predictions = np.arange(len(y_true))
            names = ["SCC", "NSCC_Adeno", "NSCC_Squamous", "Normal"]

            y_pred = np.where(scores_pred > 0.5, 1, 0)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(4):
                  fpr[i], tpr[i], _ = roc_curve(y_true[arange_like_predictions%4 == 0 + i],
                                                scores_pred[arange_like_predictions%4 == 0 + i])
                  roc_auc[i] = auc(fpr[i], tpr[i])
                  
                  print(scores_pred[arange_like_predictions%4 == 0 + i])
                  print(y_pred[arange_like_predictions%4 == 0 + i])
                  
                  accuracy = accuracy_score(y_true[arange_like_predictions%4 == 0 + i],
                                          y_pred[arange_like_predictions%4 == 0 + i])

                  print("== Final Metrics ==")
                  print(f"Accuracy {names[i]} = {accuracy:0.2f}")

                  bma = balanced_accuracy_score(y_true[arange_like_predictions%4 == 0 + i],
                                                y_pred[arange_like_predictions%4 == 0 + i])
                  kappa = cohen_kappa_score(y_true[arange_like_predictions%4 == 0 + i],
                                          y_pred[arange_like_predictions%4 == 0 + i])
                  print(f"BMA {names[i]} = {bma:0.2f}")
                  print(f"Kappa {names[i]} = {kappa:0.2f}")

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true, scores_pred)
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Plot ROC curve
            custom_params = {"axes.spines.right": False, "axes.spines.top": False}
            sns.set_theme(style="whitegrid", rc=custom_params)
            plt.figure()
            plt.plot(fpr["micro"], tpr["micro"],
                  label='micro-average ROC curve (area = {0:0.2f})'
                        ''.format(roc_auc["micro"]))
            for i in range(4):
                  plt.plot(fpr[i], tpr[i], label=f"ROC curve {names[i]} (AUC = {roc_auc[i]:0.2f})")

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate (Fpr)")
            plt.ylabel("True Positive Rate (Tpr)")
            plt.title("Validation ROC Lung Cancer Multiclass")
            plt.legend(loc="lower right")
            plt.savefig(outputdir / f"valid_{epoch + 1}_roc.svg")
            plt.clf()

elif dataset == "train":
      # Train
      outputdir = Path(bestdir / "train")
      Path(outputdir).mkdir(exist_ok=True, parents=True)

      filenames_wsis = []
      pred_scc = []
      pred_nscc_adeno = []
      pred_nscc_squamous = []
      pred_normal = []

      y_pred = []
      y_true = []
      scores_pred = []

      iterations_train= int(len(train_dataset))
      dataloader_iterator = iter(train_generator_bag)
      with torch.no_grad():
            for i in range(iterations_train):
                  print(f"Iterations: {i + 1} / {iterations_train}")
                  try:
                        wsi_id, labels = next(dataloader_iterator)
                  except StopIteration:
                        dataloader_iterator = iter(train_generator_bag)
                        wsi_id, labels = next(dataloader_iterator)
                        #inputs: bags, labels: labels of the bags

                  wsi_id = wsi_id[0]

                  labels = torch.stack(labels)
                  labels_np = labels.cpu().numpy().flatten()

                  labels_local = labels.float().flatten().to(device, non_blocking=True)

                  train_generator_instance = get_generator_instances(patches_train[wsi_id], 
                                                                  preprocess,
                                                                  cfg.dataloader.batch_size, 
                                                                  None,
                                                                  cfg.dataloader.num_workers) 

                  n_elems = len(patches_train[wsi_id])   

                  features = []
                  
                  for instances in train_generator_instance:
                        instances = instances.to(device, non_blocking=True)

                        # forward + backward + optimize
                        feats = net.conv_layers(instances)
                        feats = feats.view(-1, net.fc_input_features)
                        feats_np = feats.cpu().data.numpy()

                        features.extend(feats_np)

                              #del instances
                        #del instances
                  features_np = np.reshape(features,(n_elems, net.fc_input_features))

                  inputs = torch.tensor(features_np).float().to(device, non_blocking=True)
                  
                  logits_img, _ = net(None, inputs)


                  sigmoid_output_img = F.sigmoid(logits_img)
                  outputs_wsi_np_img = sigmoid_output_img.cpu().numpy()

                  print(f"pred_img: {outputs_wsi_np_img}")

                  filenames_wsis.append(wsi_id)
                  pred_scc.append(outputs_wsi_np_img[0])
                  pred_nscc_adeno.append(outputs_wsi_np_img[1])
                  pred_nscc_squamous.append(outputs_wsi_np_img[2])
                  pred_normal.append(outputs_wsi_np_img[3])

                  output_norm = np.where(outputs_wsi_np_img > 0.5, 1, 0)

                  y_pred = np.append(y_pred, output_norm)
                  y_true = np.append(y_true, labels_np)
                  scores_pred= np.append(scores_pred, outputs_wsi_np_img)
                  
                  accuracy_train = accuracy_score(y_true, y_pred)
                  print(f"micro_accuracy train {accuracy_train}") 

            File = {'filenames': filenames_wsis,
            'pred_scc': pred_scc, 
            'pred_nscc_adeno': pred_nscc_adeno,
            'pred_nscc_squamous': pred_nscc_squamous, 
            'pred_normal': pred_normal}

            df_predictions = pd.DataFrame.from_dict(File)
            
            filename_train_predictions = Path(outputdir / f"train_predictions_{epoch + 1}.csv")
            df_predictions.to_csv(filename_train_predictions) 

            arange_like_predictions = np.arange(len(y_true))
            names = ["SCC", "NSCC_Adeno", "NSCC_Squamous", "Normal"]

            y_pred = np.where(scores_pred > 0.5, 1, 0)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(4):
                  fpr[i], tpr[i], _ = roc_curve(y_true[arange_like_predictions%4 == 0 + i],
                                                scores_pred[arange_like_predictions%4 == 0 + i])
                  roc_auc[i] = auc(fpr[i], tpr[i])
                  
                  print(scores_pred[arange_like_predictions%4 == 0 + i])
                  print(y_pred[arange_like_predictions%4 == 0 + i])
                  
                  accuracy = accuracy_score(y_true[arange_like_predictions%4 == 0 + i],
                                          y_pred[arange_like_predictions%4 == 0 + i])

                  print("== Final Metrics ==")
                  print(f"Accuracy {names[i]} = {accuracy:0.2f}")

                  bma = balanced_accuracy_score(y_true[arange_like_predictions%4 == 0 + i],
                                                y_pred[arange_like_predictions%4 == 0 + i])
                  kappa = cohen_kappa_score(y_true[arange_like_predictions%4 == 0 + i],
                                          y_pred[arange_like_predictions%4 == 0 + i])
                  print(f"BMA {names[i]} = {bma:0.2f}")
                  print(f"Kappa {names[i]} = {kappa:0.2f}")

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true, scores_pred)
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Plot ROC curve
            custom_params = {"axes.spines.right": False, "axes.spines.top": False}
            sns.set_theme(style="whitegrid", rc=custom_params)
            plt.figure()
            plt.plot(fpr["micro"], tpr["micro"],
                  label='micro-average ROC curve (area = {0:0.2f})'
                        ''.format(roc_auc["micro"]))
            for i in range(4):
                  plt.plot(fpr[i], tpr[i], label=f"ROC curve {names[i]} (AUC = {roc_auc[i]:0.2f})")

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate (Fpr)")
            plt.ylabel("True Positive Rate (Tpr)")
            plt.title("Train ROC Lung Cancer Multiclass")
            plt.legend(loc="lower right")
            plt.savefig(outputdir / f"train_{epoch + 1}_roc.svg")
            plt.clf()
