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
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, hamming_loss, zero_one_loss
from natsort import natsorted
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import click

thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.option(
    "--experiment_name",
    default="MIL_resnet101_00025_2102030",
    prompt="Name of the MIL experiment name to compute metrics",
    help="Name of the MIL experiment name to compute metrics",
)
@click.option(
    "--dataset",
    default="test",
    prompt="Name of the dataset 'test', 'valid' or 'train'",
    help="Name of the dataset 'test', 'valid' or 'train'",
)
def main(experiment_name, dataset):

      modeldir = Path(thispath.parent.parent / "trained_models" / "MIL" / experiment_name)

      cfg = yaml_load(modeldir / f"config_{experiment_name}.yml")

      bestdir = Path(modeldir / cfg.dataset.magnification / cfg.model.model_name)

      # For logging
      logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                        encoding='utf-8',
                        level=logging.INFO,
                        handlers=[
                              logging.FileHandler(bestdir / "debug_metrics.log"),
                              logging.StreamHandler()
                        ],
                        datefmt='%m/%d/%Y %I:%M:%S %p')

      checkpoint = torch.load(bestdir / f"{experiment_name}.pt")

      train_loss = checkpoint["train_loss"]
      valid_loss = checkpoint["valid_loss"]
      epoch = checkpoint["epoch"] + 1

      logging.info(f"Loaded {experiment_name} using as backbone {cfg.model.model_name}, a Best "
                   f"Loss in Train of {train_loss} and in Validation of {valid_loss} at Epoch "
                   f"{epoch+1}.")
      logging.info("")

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

      # Loading Data Split
      k = 5

      data_split = pd.read_csv(Path(datadir / f"{k}_fold_crossvalidation_data_split.csv"),
                               index_col=0)
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

      pyhistdir = Path(datadir / "Mask_PyHIST_v2")
      
      # Discard WSI with less than 10 patches
      testdir = Path(pyhistdir / "Lung")
      metadata_test = pd.read_csv(testdir / "metadata_slides.csv", index_col=0)

      discard_wsi_test = []
      if (metadata_test['number_filtered_patches'] < 10).any():
            for index, row in metadata_test.iterrows():
                  if row['number_filtered_patches'] < 10:
                        discard_wsi_test.append(index)

            logging.info(f"There is {len(discard_wsi_test)} WSI discarded in test, <10 patches")
            logging.info(discard_wsi_test)

      # Load Test Dataset and Labels
      test_csv = pd.read_csv(Path(datadir / f"manual_labels_test.csv"), index_col=0)
      test_csv.index = test_csv.index.str.replace("/", '-')
      test_csv.drop(discard_wsi_test, inplace=True)

      test_dataset = test_csv.index
      test_labels = test_csv.values

      # Load Test patches
      if dataset == "test":
            
            dataset_path = natsorted([i for i in testdir.rglob("*_densely_filtered_paths_v2.csv")])

            patches_test = {}
            for wsi_patches_path in tqdm(dataset_path, desc="Selecting patches: "):

                  csv_patch_path = pd.read_csv(wsi_patches_path).to_numpy()
                  
                  name = wsi_patches_path.parent.stem
                  patches_test[name] = csv_patch_path

            for discard_wsi in discard_wsi_test:
                  patches_test.pop(discard_wsi, None)

            logging.info(f"Total number of WSI for test {len(patches_test)}")
            logging.info("")

            # Load datasets
            batch_size_bag = cfg.dataloader.batch_size_bag
            
            params_test_bag = {'batch_size': batch_size_bag,
                              'shuffle': False}

            test_set_bag = Dataset_bag_MIL(test_dataset, test_labels)
            test_generator_bag = DataLoader(test_set_bag, **params_test_bag)

      else:
            dataset_path = natsorted([i for i in pyhistdir.rglob("*_densely_filtered_paths.csv")
                                      if "LungAOEC" in str(i)])

            patches_path = {}
            for wsi_patches_path in tqdm(dataset_path, desc="Selecting patches: "):

                  csv_patch_path = pd.read_csv(wsi_patches_path).to_numpy()
                  
                  name = wsi_patches_path.parent.stem
                  patches_path[name] = csv_patch_path

            patches_train = {}
            patches_validation = {}
            
            for value, key in zip(patches_path.values(), patches_path.keys()):
                  if key in train_dataset:
                        patches_train[key] = value
                  if key in validation_dataset:
                        patches_validation[key] = value
                  if key in test_dataset:
                        patches_test[key] = value

            logging.info(f"Total number of WSI for train {len(patches_train.values())}")
            logging.info(f"Total number of WSI for validation {len(patches_validation.values())}")
            logging.info("")

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

      # Data normalization
      preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev),
            transforms.Resize(size=(model.resize_param, model.resize_param),
            antialias=True)
      ])

      # Train and validation metrics from saved predictions
      labels = pd.read_csv(datadir / "labels.csv", index_col=0)
      train_predictions_df = pd.read_csv(bestdir / "training_predictions_best.csv", index_col="filenames")
      train_predictions_df.sort_index(inplace=True)
      train_predictions_df.drop(columns=train_predictions_df.columns[0], axis=1, inplace=True)
      train_predictions_df = train_predictions_df[~train_predictions_df.index.duplicated(keep='first')]
      train_pred_scores = train_predictions_df.values
      train_pred = np.where(train_pred_scores > 0.5, 1, 0)

      valid_predictions_df = pd.read_csv(bestdir / "validation_predictions_best.csv", index_col="filenames")
      valid_predictions_df.sort_index(inplace=True)
      valid_predictions_df.drop(columns=valid_predictions_df.columns[0], axis=1, inplace=True)
      valid_predictions_df = valid_predictions_df[~valid_predictions_df.index.duplicated(keep='first')]
      valid_pred_scores = valid_predictions_df.values
      valid_pred = np.where(valid_pred_scores > 0.5, 1, 0)

      wsi_train = labels.index.intersection(train_predictions_df.index)
      labels_train = labels.loc[wsi_train].values
      wsi_valid = labels.index.intersection(valid_predictions_df.index)
      labels_valid = labels.loc[wsi_valid].values

      labels_train_flat = labels_train.flatten()
      train_pred_flat = train_pred.flatten()
      train_scores_flat = train_pred_scores.flatten()
      labels_valid_flat = labels_valid.flatten()
      valid_pred_flat = valid_pred.flatten()
      valid_scores_flat = valid_pred_scores.flatten()

      arange_like_train = np.arange(len(labels_train_flat))
      arange_like_valid = np.arange(len(labels_valid_flat))

      logging.info("== Final Metrics Train ==")
      names = ["SCC", "NSCC Adeno", "NSCC Squamous", "No Cancer"]
      for i in range(4):
            accuracy_train = accuracy_score(labels_train_flat[arange_like_train%4 == 0 + i],
                                            train_pred_flat[arange_like_train%4 == 0 + i])
            bma_train = balanced_accuracy_score(labels_train_flat[arange_like_train%4 == 0 + i],
                                                train_pred_flat[arange_like_train%4 == 0 + i])
            kappa_train = cohen_kappa_score(labels_train_flat[arange_like_train%4 == 0 + i],
                                            train_pred_flat[arange_like_train%4 == 0 + i])

            
            logging.info(f"Accuracy {names[i]} = {accuracy_train:0.2f}")
            logging.info(f"BMA {names[i]} = {bma_train:0.2f}")
            logging.info(f"Kappa {names[i]} = {kappa_train:0.2f}")
            logging.info("")

      
      f1_micro_train = f1_score(train_pred, labels_train, average="micro")
      f1_macro_train = f1_score(train_pred, labels_train, average="macro")
      f1_weighted_train = f1_score(train_pred, labels_train, average="weighted")

      logging.info(f"f1 Score Micro average in train = {f1_micro_train:0.2f}")
      logging.info(f"f1 Score Macro average in train = {f1_macro_train:0.2f}")
      logging.info(f"f1 Score Weighted average in train = {f1_weighted_train:0.2f}")
      logging.info("")

      hamming_train = hamming_loss(train_pred, labels_train)
      zero_one_train = zero_one_loss(train_pred, labels_train)
      logging.info(f"Hamming loss in train = {hamming_train:0.2f}")
      logging.info(f"0-1 loss in train = {zero_one_train:0.2f}")
      logging.info("")

      logging.info("== Final Metrics Validation ==")
      for i in range(4):
            accuracy_valid = accuracy_score(labels_valid_flat[arange_like_valid%4 == 0 + i],
                                            valid_pred_flat[arange_like_valid%4 == 0 + i])
            bma_valid = balanced_accuracy_score(labels_valid_flat[arange_like_valid%4 == 0 + i],
                                                valid_pred_flat[arange_like_valid%4 == 0 + i])
            kappa_valid = cohen_kappa_score(labels_valid_flat[arange_like_valid%4 == 0 + i],
                                            valid_pred_flat[arange_like_valid%4 == 0 + i])

            
            logging.info(f"Accuracy {names[i]} = {accuracy_valid:0.2f}")
            logging.info(f"BMA {names[i]} = {bma_valid:0.2f}")
            logging.info(f"Kappa {names[i]} = {kappa_valid:0.2f}")
            logging.info("")

      f1_micro_valid = f1_score(valid_pred, labels_valid, average="micro")
      f1_macro_valid = f1_score(valid_pred, labels_valid, average="macro")
      f1_weighted_valid = f1_score(valid_pred, labels_valid, average="weighted")

      logging.info(f"f1 Score Micro average in validation = {f1_micro_valid:0.2f}")
      logging.info(f"f1 Score Macro average in validation = {f1_macro_valid:0.2f}")
      logging.info(f"f1 Score Weighted average in validation = {f1_weighted_valid:0.2f}")
      logging.info("")

      hamming_valid = hamming_loss(valid_pred, labels_valid)
      zero_one_valid = zero_one_loss(valid_pred, labels_valid)
      logging.info(f"Hamming loss in validation = {hamming_valid:0.2f}")
      logging.info(f"0-1 loss in validation = {zero_one_valid:0.2f}")
      logging.info("")
      
      # Multilabel Confusion Matrix
      outputdir_train = Path(bestdir / "train_metrics")
      Path(outputdir_train).mkdir(exist_ok=True, parents=True)
      outputdir_valid = Path(bestdir / "valid_metrics")
      Path(outputdir_valid).mkdir(exist_ok=True, parents=True)

      confusion_matrix_train = multilabel_confusion_matrix(labels_train, train_pred)

      plt.figure()
      disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_train[0, :, :],
                                    display_labels=[False, True])
      disp.plot()
      plt.title(f"Confusion Matrix ({names[0]})")
      plt.savefig(outputdir_train / f"train_{epoch + 1}_cm_{names[0]}.svg")
      plt.close()

      plt.figure()
      disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_train[1, :, :],
                                    display_labels=[False, True])
      disp.plot()
      plt.title(f"Confusion Matrix ({names[1]})")
      plt.savefig(outputdir_train / f"train_{epoch + 1}_cm_{names[1]}.svg")
      plt.close()

      plt.figure()
      disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_train[2, :, :],
                                    display_labels=[False, True])
      disp.plot()
      plt.title(f"Confusion Matrix ({names[2]})")
      plt.savefig(outputdir_train / f"train_{epoch + 1}_cm_{names[2]}.svg")
      plt.close()

      plt.figure()
      disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_train[3, :, :],
                                    display_labels=[False, True])
      disp.plot()
      plt.title(f"Confusion Matrix ({names[3]})")
      plt.savefig(outputdir_train / f"train_{epoch + 1}_cm_{names[3]}.svg")
      plt.close()

      # Compute ROC curve (AUC) and PR curve (AP)
      fpr_train = {}
      tpr_train = {}
      roc_auc_train = {}
      precision_train = {}
      recall_train = {}
      avg_precision_train = {}
      for i in range(4):
            fpr_train[i], tpr_train[i], _ = roc_curve(
                                          labels_train_flat[arange_like_train%4 == 0 + i],
                                          train_scores_flat[arange_like_train%4 == 0 + i])
            roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])

            precision_train[i], recall_train[i], _ = precision_recall_curve(
                                                labels_train_flat[arange_like_train%4 == 0 + i],
                                                train_scores_flat[arange_like_train%4 == 0 + i])
            avg_precision_train[i] = average_precision_score(
                                                labels_train_flat[arange_like_train%4 == 0 + i],
                                                train_scores_flat[arange_like_train%4 == 0 + i])


      # Compute micro-average ROC curve and ROC area
      fpr_train["micro"], tpr_train["micro"], _ = roc_curve(labels_train_flat, train_scores_flat)
      roc_auc_train["micro"] = auc(fpr_train["micro"], tpr_train["micro"])
      
      precision_train["micro"], recall_train["micro"], _ = precision_recall_curve(labels_train_flat,
                                                                      train_scores_flat)
      avg_precision_train["micro"] = average_precision_score(labels_train,
                                                             train_pred_scores,
                                                             average="micro")

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
      plt.savefig(outputdir_train / f"train_{epoch + 1}_roc.svg")
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
      plt.savefig(outputdir_train / f"train_{epoch + 1}_pr_curve.svg")
      plt.close()

      confusion_matrix_valid = multilabel_confusion_matrix(labels_valid, valid_pred)

      plt.figure()
      disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_valid[0, :, :],
                                    display_labels=[False, True])
      disp.plot()
      plt.title(f"Confusion Matrix ({names[0]})")
      plt.savefig(outputdir_valid / f"valid_{epoch + 1}_cm_{names[0]}.svg")
      plt.close()

      plt.figure()
      disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_valid[1, :, :],
                                    display_labels=[False, True])
      disp.plot()
      plt.title(f"Confusion Matrix ({names[1]})")
      plt.savefig(outputdir_valid / f"valid_{epoch + 1}_cm_{names[1]}.svg")
      plt.close()

      plt.figure()
      disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_valid[2, :, :],
                                    display_labels=[False, True])
      disp.plot()
      plt.title(f"Confusion Matrix ({names[2]})")
      plt.savefig(outputdir_valid / f"valid_{epoch + 1}_cm_{names[2]}.svg")
      plt.close()

      plt.figure()
      disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_valid[3, :, :],
                                    display_labels=[False, True])
      disp.plot()
      plt.title(f"Confusion Matrix ({names[3]})")
      plt.savefig(outputdir_valid / f"valid_{epoch + 1}_cm_{names[3]}.svg")
      plt.close()

      # Compute ROC curve (AUC) and PR curve (AP)
      fpr_valid = {}
      tpr_valid = {}
      roc_auc_valid = {}
      precision_valid = {}
      recall_valid = {}
      avg_precision_valid = {}
      for i in range(4):
            fpr_valid[i], tpr_valid[i], _ = roc_curve(
                                          labels_valid_flat[arange_like_valid%4 == 0 + i],
                                          valid_scores_flat[arange_like_valid%4 == 0 + i])
            roc_auc_valid[i] = auc(fpr_valid[i], tpr_valid[i])

            precision_valid[i], recall_valid[i], _ = precision_recall_curve(
                                                labels_valid_flat[arange_like_valid%4 == 0 + i],
                                                valid_scores_flat[arange_like_valid%4 == 0 + i])
            avg_precision_valid[i] = average_precision_score(
                                                labels_valid_flat[arange_like_valid%4 == 0 + i],
                                                valid_scores_flat[arange_like_valid%4 == 0 + i])


      # Compute micro-average ROC curve and ROC area
      fpr_valid["micro"], tpr_valid["micro"], _ = roc_curve(labels_valid_flat, valid_scores_flat)
      roc_auc_valid["micro"] = auc(fpr_valid["micro"], tpr_valid["micro"])
      
      precision_valid["micro"], recall_valid["micro"], _ = precision_recall_curve(labels_valid_flat,
                                                                      valid_scores_flat)
      avg_precision_valid["micro"] = average_precision_score(labels_valid,
                                                             valid_pred_scores,
                                                             average="micro")

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
      plt.savefig(outputdir_valid / f"valid_{epoch + 1}_roc.svg")
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
      plt.savefig(outputdir_valid / f"valid_{epoch + 1}_pr_curve.svg")
      plt.close()

      logging.info(f"Plot metrics of train and validation save on {bestdir}")

      if dataset == "test":
            # Test
            outputdir = Path(bestdir / "test_metrics")
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
                        logging.info(f"Iterations: {i + 1} / {iterations_test}")
                        try:
                              wsi_id, labels = next(dataloader_iterator)
                        except StopIteration:
                              dataloader_iterator = iter(test_generator_bag)
                              wsi_id, labels = next(dataloader_iterator)
                              #inputs: bags, labels: labels of the bags

                        wsi_id = wsi_id[0]
                        
                        labels_np = labels.cpu().numpy().flatten()

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

                        logging.info(f"pred_img_logits: {outputs_wsi_np_img}")

                        filenames_wsis.append(wsi_id)
                        pred_scc.append(outputs_wsi_np_img[0])
                        pred_nscc_adeno.append(outputs_wsi_np_img[1])
                        pred_nscc_squamous.append(outputs_wsi_np_img[2])
                        pred_normal.append(outputs_wsi_np_img[3])

                        output_norm = np.where(outputs_wsi_np_img > 0.5, 1, 0)

                        if np.all(output_norm == 0):
                              label = np.argmax(outputs_wsi_np_img)
                              output_norm[label] = 1

                        logging.info(f"pred_img: {output_norm}")
                        logging.info(f"y_true: {labels_np}")

                        y_pred = np.append(y_pred, output_norm)
                        y_true = np.append(y_true, labels_np)
                        scores_pred= np.append(scores_pred, outputs_wsi_np_img)
                        
                        micro_accuracy_test = accuracy_score(y_true, y_pred)
                        logging.info(f"Accuracy test: {micro_accuracy_test}") 

                  File = {'filenames': filenames_wsis,
                  'pred_scc': pred_scc, 
                  'pred_nscc_adeno': pred_nscc_adeno,
                  'pred_nscc_squamous': pred_nscc_squamous, 
                  'pred_normal': pred_normal}

                  df_predictions = pd.DataFrame.from_dict(File)
                  
                  filename_test_predictions = Path(outputdir / f"test_predictions_{epoch + 1}.csv")
                  df_predictions.to_csv(filename_test_predictions) 

                  arange_like_predictions = np.arange(len(y_true))
                  
                  y_pred_reshape = y_pred.reshape(len(test_dataset), 4)
                  y_true_reshape = y_true.reshape(len(test_dataset), 4)
                  scores_pred_reshape = scores_pred.reshape(len(test_dataset), 4)

                  confusion_matrix = multilabel_confusion_matrix(y_true_reshape, y_pred_reshape)

                  plt.figure()
                  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[0, :, :],
                                                display_labels=[False, True])
                  disp.plot()
                  plt.title(f"Confusion Matrix ({names[0]})")
                  plt.savefig(outputdir / f"test_{epoch + 1}_cm_{names[0]}.svg")
                  plt.close()

                  plt.figure()
                  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[1, :, :],
                                                display_labels=[False, True])
                  disp.plot()
                  plt.title(f"Confusion Matrix ({names[1]})")
                  plt.savefig(outputdir / f"test_{epoch + 1}_cm_{names[1]}.svg")
                  plt.close()

                  plt.figure()
                  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[2, :, :],
                                                display_labels=[False, True])
                  disp.plot()
                  plt.title(f"Confusion Matrix ({names[2]})")
                  plt.savefig(outputdir / f"test_{epoch + 1}_cm_{names[2]}.svg")
                  plt.close()

                  plt.figure()
                  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[3, :, :],
                                                display_labels=[False, True])
                  disp.plot()
                  plt.title(f"Confusion Matrix ({names[3]})")
                  plt.savefig(outputdir / f"test_{epoch + 1}_cm_{names[3]}.svg")
                  plt.close()

                  # Compute ROC curve and ROC area for each class
                  fpr = dict()
                  tpr = dict()
                  precision = {}
                  recall = {}
                  avg_precision = {}
                  roc_auc = dict()

                  logging.info("")
                  logging.info("== Final Metrics Test ==")
                  for i in range(4):
                        fpr[i], tpr[i], _ = roc_curve(
                                                y_true[arange_like_predictions%4 == 0 + i],
                                                scores_pred[arange_like_predictions%4 == 0 + i])
                        roc_auc[i] = auc(fpr[i], tpr[i])

                        precision[i], recall[i], _ = precision_recall_curve(
                                                      y_true[arange_like_predictions%4 == 0 + i],
                                                      scores_pred[arange_like_predictions%4 == 0 + i])
                        avg_precision[i] = average_precision_score(
                                                      y_true[arange_like_predictions%4 == 0 + i],
                                                      scores_pred[arange_like_predictions%4 == 0 + i])


                        accuracy = accuracy_score(y_true[arange_like_predictions%4 == 0 + i],
                                                y_pred[arange_like_predictions%4 == 0 + i])

                        logging.info(f"Accuracy {names[i]} = {accuracy:0.2f}")

                        bma = balanced_accuracy_score(y_true[arange_like_predictions%4 == 0 + i],
                                                      y_pred[arange_like_predictions%4 == 0 + i])
                        kappa = cohen_kappa_score(y_true[arange_like_predictions%4 == 0 + i],
                                                y_pred[arange_like_predictions%4 == 0 + i])
                        logging.info(f"BMA {names[i]} = {bma:0.2f}")
                        logging.info(f"Kappa {names[i]} = {kappa:0.2f}")
                        logging.info("")
                  
                  f1_micro_test = f1_score(y_pred_reshape, y_true_reshape, average="micro")
                  f1_macro_test = f1_score(y_pred_reshape, y_true_reshape, average="macro")
                  f1_weighted_test = f1_score(y_pred_reshape, y_true_reshape, average="weighted")

                  logging.info(f"f1 Score Micro average in test = {f1_micro_test:0.2f}")
                  logging.info(f"f1 Score Macro average in test = {f1_macro_test:0.2f}")
                  logging.info(f"f1 Score Weighted average in test = {f1_weighted_test:0.2f}")
                  logging.info("")

                  hamming_test = hamming_loss(y_pred, y_true)
                  zero_one_test = zero_one_loss(y_pred, y_true)
                  logging.info(f"Hamming loss in test = {hamming_test:0.2f}")
                  logging.info(f"0-1 loss in test = {zero_one_test:0.2f}")
                  logging.info("")

                  # Compute micro-average ROC curve and ROC area
                  fpr["micro"], tpr["micro"], _ = roc_curve(y_true, scores_pred)
                  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                  
                  precision["micro"], recall["micro"], _ = precision_recall_curve(y_true,
                                                                                  scores_pred)
                  avg_precision["micro"] = average_precision_score(y_true_reshape,
                                                                  scores_pred_reshape,
                                                                  average="micro")

                  # Plot ROC curve
                  custom_params = {"axes.spines.right": False, "axes.spines.top": False}
                  sns.set_theme(style="whitegrid", rc=custom_params)
                  plt.figure()
                  plt.plot(fpr["micro"], tpr["micro"],
                        label='micro-average ROC curve (AUC = {0:0.2f})'
                              ''.format(roc_auc["micro"]))
                  for i in range(4):
                        plt.plot(fpr[i],
                              tpr[i],
                              label=f"ROC curve {names[i]} (AUC = {roc_auc[i]:0.2f})")

                  plt.plot([0, 1], [0, 1], 'k--')
                  plt.xlim([0.0, 1.0])
                  plt.ylim([0.0, 1.05])
                  plt.xlabel("False Positive Rate (Fpr)")
                  plt.ylabel("True Positive Rate (Tpr)")
                  plt.title("Test ROC Lung Cancer Multiclass")
                  plt.legend(loc="lower right")
                  plt.savefig(outputdir / f"test_{epoch + 1}_roc.svg")
                  plt.close()


                  sns.set_theme(style="white", rc=custom_params)
                  plt.figure()
                  f_scores = np.linspace(0.2, 0.8, num=4)
                  for f_score in f_scores:
                        x = np.linspace(0.01, 1)
                        y = f_score * x / (2 * x - f_score)
                        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
                        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

                  plt.plot(recall["micro"], precision["micro"],
                        label='micro-average PR curve (AP = {0:0.2f})'
                              ''.format(avg_precision["micro"]))
                  for i in range(4):
                        plt.plot(recall[i],
                              precision[i],
                              label=f"PR curve {names[i]} (AP = {avg_precision[i]:0.2f})")

                  plt.xlim([0.0, 1.0])
                  plt.ylim([0.0, 1.05])
                  plt.xlabel("Recall")
                  plt.ylabel("Precision")
                  plt.title("Test PR curve Lung Cancer Multiclass")
                  plt.legend(loc="lower left")
                  plt.savefig(outputdir / f"test_{epoch + 1}_pr_curve.svg")
                  plt.close()

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
                        logging.info(f"Iterations: {i + 1} / {iterations_valid}")
                        try:
                              wsi_id, labels = next(dataloader_iterator)
                        except StopIteration:
                              dataloader_iterator = iter(valid_generator_bag)
                              wsi_id, labels = next(dataloader_iterator)
                              #inputs: bags, labels: labels of the bags

                        wsi_id = wsi_id[0]
                        labels = torch.stack(labels)
                        labels_np = labels.cpu().numpy().flatten()

                        valid_generator_instance = get_generator_instances(
                                                                        patches_validation[wsi_id], 
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

                        logging.info(f"pred_img: {outputs_wsi_np_img}")

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
                        logging.info(f"micro_accuracy validation {accuracy_valid}") 

                  File = {'filenames': filenames_wsis,
                  'pred_scc': pred_scc, 
                  'pred_nscc_adeno': pred_nscc_adeno,
                  'pred_nscc_squamous': pred_nscc_squamous, 
                  'pred_normal': pred_normal}

                  df_predictions = pd.DataFrame.from_dict(File)
                  
                  filename_valid_predictions = Path(outputdir /
                                                    f"valid_predictions_{epoch + 1}.csv")
                  df_predictions.to_csv(filename_valid_predictions) 

                  arange_like_predictions = np.arange(len(y_true))

                  y_pred_reshape = y_pred.reshape(len(validation_dataset), 4)
                  y_true_reshape = y_true.reshape(len(validation_dataset), 4)
                  scores_pred_reshape = scores_pred.reshape(len(validation_dataset), 4)

                  confusion_matrix = multilabel_confusion_matrix(y_true_reshape, y_pred_reshape)

                  plt.figure()
                  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[0, :, :],
                                                display_labels=[False, True])
                  disp.plot()
                  plt.title(f"Confusion Matrix ({names[0]})")
                  plt.savefig(outputdir / f"valid_{epoch + 1}_cm_{names[0]}.svg")
                  plt.close()

                  plt.figure()
                  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[1, :, :],
                                                display_labels=[False, True])
                  disp.plot()
                  plt.title(f"Confusion Matrix ({names[1]})")
                  plt.savefig(outputdir / f"valid_{epoch + 1}_cm_{names[1]}.svg")
                  plt.close()

                  plt.figure()
                  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[2, :, :],
                                                display_labels=[False, True])
                  disp.plot()
                  plt.title(f"Confusion Matrix ({names[2]})")
                  plt.savefig(outputdir / f"valid_{epoch + 1}_cm_{names[2]}.svg")
                  plt.close()

                  plt.figure()
                  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[3, :, :],
                                                display_labels=[False, True])
                  disp.plot()
                  plt.title(f"Confusion Matrix ({names[3]})")
                  plt.savefig(outputdir / f"valid_{epoch + 1}_cm_{names[3]}.svg")
                  plt.close()

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


                        accuracy = accuracy_score(y_true[arange_like_predictions%4 == 0 + i],
                                                y_pred[arange_like_predictions%4 == 0 + i])

                        logging.info("== Final Metrics  Validation ==")
                        logging.info(f"Accuracy {names[i]} = {accuracy:0.2f}")

                        bma = balanced_accuracy_score(y_true[arange_like_predictions%4 == 0 + i],
                                                      y_pred[arange_like_predictions%4 == 0 + i])
                        kappa = cohen_kappa_score(y_true[arange_like_predictions%4 == 0 + i],
                                                y_pred[arange_like_predictions%4 == 0 + i])
                        logging.info(f"BMA {names[i]} = {bma:0.2f}")
                        logging.info(f"Kappa {names[i]} = {kappa:0.2f}")

                  # Compute micro-average ROC curve and ROC area
                  fpr["micro"], tpr["micro"], _ = roc_curve(y_true, scores_pred)
                  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                  
                  precision["micro"], recall["micro"], _ = precision_recall_curve(y_true, scores_pred)
                  avg_precision["micro"] = average_precision_score(y_true_reshape,
                                                                  scores_pred_reshape,
                                                                  average="micro")

                  # Plot ROC curve
                  custom_params = {"axes.spines.right": False, "axes.spines.top": False}
                  sns.set_theme(style="whitegrid", rc=custom_params)
                  plt.figure()
                  plt.plot(fpr["micro"], tpr["micro"],
                        label='micro-average ROC curve (AUC = {0:0.2f})'
                              ''.format(roc_auc["micro"]))
                  for i in range(4):
                        plt.plot(fpr[i],
                              tpr[i],
                              label=f"ROC curve {names[i]} (AUC = {roc_auc[i]:0.2f})")

                  plt.plot([0, 1], [0, 1], 'k--')
                  plt.xlim([0.0, 1.0])
                  plt.ylim([0.0, 1.05])
                  plt.xlabel("False Positive Rate (Fpr)")
                  plt.ylabel("True Positive Rate (Tpr)")
                  plt.title("Validation ROC Lung Cancer Multiclass")
                  plt.legend(loc="lower right")
                  plt.savefig(outputdir / f"valid_{epoch + 1}_roc.svg")
                  plt.close()


                  sns.set_theme(style="white", rc=custom_params)
                  plt.figure()
                  f_scores = np.linspace(0.2, 0.8, num=4)
                  for f_score in f_scores:
                        x = np.linspace(0.01, 1)
                        y = f_score * x / (2 * x - f_score)
                        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
                        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

                  plt.plot(recall["micro"], precision["micro"],
                        label='micro-average PR curve (AP = {0:0.2f})'
                              ''.format(avg_precision["micro"]))
                  for i in range(4):
                        plt.plot(recall[i],
                              precision[i],
                              label=f"PR curve {names[i]} (AP = {avg_precision[i]:0.2f})")

                  plt.plot([0, 1], [0, 1], 'k--')
                  plt.xlim([0.0, 1.0])
                  plt.ylim([0.0, 1.05])
                  plt.xlabel("Recall")
                  plt.ylabel("Precision")
                  plt.title("Validation PR curve Lung Cancer Multiclass")
                  plt.legend(loc="lower left")
                  plt.savefig(outputdir / f"valid_{epoch + 1}_pr_curve.svg")
                  plt.close()

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
                        logging.info(f"Iterations: {i + 1} / {iterations_train}")
                        try:
                              wsi_id, labels = next(dataloader_iterator)
                        except StopIteration:
                              dataloader_iterator = iter(train_generator_bag)
                              wsi_id, labels = next(dataloader_iterator)
                              #inputs: bags, labels: labels of the bags

                        wsi_id = wsi_id[0]

                        labels = torch.stack(labels)
                        labels_np = labels.cpu().numpy().flatten()

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

                        logging.info(f"pred_img: {outputs_wsi_np_img}")

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
                        logging.info(f"micro_accuracy train {accuracy_train}") 

                  File = {'filenames': filenames_wsis,
                  'pred_scc': pred_scc, 
                  'pred_nscc_adeno': pred_nscc_adeno,
                  'pred_nscc_squamous': pred_nscc_squamous, 
                  'pred_normal': pred_normal}

                  df_predictions = pd.DataFrame.from_dict(File)
                  
                  filename_train_predictions = Path(outputdir / f"train_predictions_{epoch + 1}.csv")
                  df_predictions.to_csv(filename_train_predictions) 

                  arange_like_predictions = np.arange(len(y_true))

                  y_pred_reshape = y_pred.reshape(len(train_dataset), 4)
                  y_true_reshape = y_true.reshape(len(train_dataset), 4)
                  scores_pred_reshape = scores_pred.reshape(len(train_dataset), 4)

                  confusion_matrix = multilabel_confusion_matrix(y_true_reshape, y_pred_reshape)

                  plt.figure()
                  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[0, :, :],
                                                display_labels=[False, True])
                  disp.plot()
                  plt.title(f"Confusion Matrix ({names[0]})")
                  plt.savefig(outputdir / f"train_{epoch + 1}_cm_{names[0]}.svg")
                  plt.close()

                  plt.figure()
                  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[1, :, :],
                                                display_labels=[False, True])
                  disp.plot()
                  plt.title(f"Confusion Matrix ({names[1]})")
                  plt.savefig(outputdir / f"train_{epoch + 1}_cm_{names[1]}.svg")
                  plt.close()

                  plt.figure()
                  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[2, :, :],
                                                display_labels=[False, True])
                  disp.plot()
                  plt.title(f"Confusion Matrix ({names[2]})")
                  plt.savefig(outputdir / f"train_{epoch + 1}_cm_{names[2]}.svg")
                  plt.close()

                  plt.figure()
                  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[3, :, :],
                                                display_labels=[False, True])
                  disp.plot()
                  plt.title(f"Confusion Matrix ({names[3]})")
                  plt.savefig(outputdir / f"train_{epoch + 1}_cm_{names[3]}.svg")
                  plt.close()

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


                        accuracy = accuracy_score(y_true[arange_like_predictions%4 == 0 + i],
                                                y_pred[arange_like_predictions%4 == 0 + i])

                        logging.info("== Final Metrics Train ==")
                        logging.info(f"Accuracy {names[i]} = {accuracy:0.2f}")

                        bma = balanced_accuracy_score(y_true[arange_like_predictions%4 == 0 + i],
                                                      y_pred[arange_like_predictions%4 == 0 + i])
                        kappa = cohen_kappa_score(y_true[arange_like_predictions%4 == 0 + i],
                                                y_pred[arange_like_predictions%4 == 0 + i])
                        logging.info(f"BMA {names[i]} = {bma:0.2f}")
                        logging.info(f"Kappa {names[i]} = {kappa:0.2f}")

                  # Compute micro-average ROC curve and ROC area
                  fpr["micro"], tpr["micro"], _ = roc_curve(y_true, scores_pred)
                  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                  
                  precision["micro"], recall["micro"], _ = precision_recall_curve(y_true, scores_pred)
                  avg_precision["micro"] = average_precision_score(y_true_reshape,
                                                                  scores_pred_reshape,
                                                                  average="micro")

                  # Plot ROC curve
                  custom_params = {"axes.spines.right": False, "axes.spines.top": False}
                  sns.set_theme(style="whitegrid", rc=custom_params)
                  plt.figure()
                  plt.plot(fpr["micro"], tpr["micro"],
                        label='micro-average ROC curve (AUC = {0:0.2f})'
                              ''.format(roc_auc["micro"]))
                  for i in range(4):
                        plt.plot(fpr[i],
                              tpr[i],
                              label=f"ROC curve {names[i]} (AUC = {roc_auc[i]:0.2f})")

                  plt.plot([0, 1], [0, 1], 'k--')
                  plt.xlim([0.0, 1.0])
                  plt.ylim([0.0, 1.05])
                  plt.xlabel("False Positive Rate (Fpr)")
                  plt.ylabel("True Positive Rate (Tpr)")
                  plt.title("Train ROC Lung Cancer Multiclass")
                  plt.legend(loc="lower right")
                  plt.savefig(outputdir / f"train_{epoch + 1}_roc.svg")
                  plt.close()


                  sns.set_theme(style="white", rc=custom_params)
                  plt.figure()
                  f_scores = np.linspace(0.2, 0.8, num=4)
                  for f_score in f_scores:
                        x = np.linspace(0.01, 1)
                        y = f_score * x / (2 * x - f_score)
                        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
                        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

                  plt.plot(recall["micro"], precision["micro"],
                        label='micro-average PR curve (AP = {0:0.2f})'
                              ''.format(avg_precision["micro"]))
                  for i in range(4):
                        plt.plot(recall[i],
                              precision[i],
                              label=f"PR curve {names[i]} (AP = {avg_precision[i]:0.2f})")

                  plt.plot([0, 1], [0, 1], 'k--')
                  plt.xlim([0.0, 1.0])
                  plt.ylim([0.0, 1.05])
                  plt.xlabel("Recall")
                  plt.ylabel("Precision")
                  plt.title("Train PR curve Lung Cancer Multiclass")
                  plt.legend(loc="lower left")
                  plt.savefig(outputdir / f"train_{epoch + 1}_pr_curve.svg")
                  plt.close()


if __name__ == '__main__':
    main()
