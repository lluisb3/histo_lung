from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import albumentations as A
import time
import tqdm
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

#from pytorch_pretrained_bert.modeling import BertModel
import pyspng

thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args = sys.argv[1:]

print("CUDA current device " + str(torch.cuda.current_device()))
print("CUDA devices available " + str(torch.cuda.device_count()))

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='resnet34')
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=512)
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=15)
parser.add_argument('-p', '--pool', help='pooling algorithm',type=str, default='att')
parser.add_argument('-t', '--TASK', help='task (binary/multilabel)',type=str, default='multilabel')
parser.add_argument('-m', '--MAG', help='magnification to select',type=str, default='10')
parser.add_argument('-f', '--features', help='features_to_use: embedding (True) or features from CNN (False)',type=str, default='True')
parser.add_argument('-z', '--ausiliary', help='loss to match embeddings: mse, cosine, l1, cosine_mse',type=str, default='semi_supervised')
parser.add_argument('-g', '--activation', help='tahn/relu',type=str, default='tanh')
parser.add_argument('-a', '--preprocessed', help='pre-processed data: True False',type=str, default='False')
parser.add_argument('-x', '--encoding', help='multiclass/multilabel',type=str, default='multilabel')
parser.add_argument('-d', '--DATA', help='data to use: main/finetune/pretraining/finetune_pretrain',type=str, default='main')
parser.add_argument('-w', '--WEIGHTS', help='pre-trained weights to use',type=str, default='moco2')
parser.add_argument('-r', '--MODALITY', help='multimodal/unimodal',type=str, default='multimodal')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
pool_algorithm = args.pool
TASK = args.TASK
MAGNIFICATION = args.MAG
EMBEDDING_bool = args.features
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
LOSS_EMBEDDINGS = args.ausiliary 
GATED_bool = False
PREPROCESSED_DATA = args.preprocessed
ENCODING = args.encoding
flag_dataset = args.DATA
FLAG_CLASSIFIER = True
pre_trained_weights = args.WEIGHTS
MODALITY = args.MODALITY
ACTIVATION = args.activation

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("PARAMETERS")
print("TASK: " + str(TASK))
print("CNN used: " + str(CNN_TO_USE))
print("POOLING ALGORITHM: " + str(pool_algorithm))
print("BATCH_SIZE: " + str(BATCH_SIZE_str))
print("MAGNIFICATION: " + str(MAGNIFICATION))

#torch.backends.cudnn.benchmark = True

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


# Read the configuration file
configdir = Path(thispath.parent / f"{config_file}.yml")
cfg = yaml_load(configdir)

# Create directory to save the resuls
outputdir = Path(thispath.parent.parent / "trained_models" / "MIL" / f"{cfg.experiment_name}")
Path(outputdir).mkdir(exist_ok=True, parents=True)

# wandb login
wandb.login()

if cfg.wandb.enable:
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

model_weights_filename_CNN = outputdir+'MIL_model_'+TASK+'.pt'
model_weights_filename_CNN_temp = outputdir+'MIL_model_'+TASK+'_temp.pt'
model_weights_filename_checkpoint = outputdir+'checkpoint.pt'

#path model file
experiment_name = "MoCo_try_Adam"

mocodir = Path(thispath.parent.parent / 
               "trained_models" / 
               "MoCo" / 
               experiment_name)

cfg_moco = yaml_load(mocodir / f"config_{experiment_name}.yml")

model_moco = ModelOption(cfg.model.model_name,
                cfg.model.num_classes,
                freeze=cfg.model.freeze_weights,
                num_freezed_layers=cfg.model.num_frozen_layers,
                dropout=cfg.model.dropout,
                embedding_bool=cfg.model.embedding_bool
                )    

# Encoder and momentum encoder
moco_dim = cfg_moco.training.moco_dim

encoder = Encoder(model_moco, dim=moco_dim).to(device)

checkpoint_moco = torch.load(Path(mocodir /
                             cfg_moco.dataset.magnification / 
                             cfg_moco.model.model_name / 
                             "MoCo.pt"))
encoder.load_state_dict(checkpoint_moco["encoder_state_dict"])
loss_moco = checkpoint_moco["loss"]
epoch_moco = checkpoint_moco["epoch"] + 1

print(f"Loaded encoder using as backbone {model_moco.model_name} with a best"
      f"loss of {loss_moco} at Epoch {epoch_moco}")

pyhistdir = Path(datadir / "Mask_PyHIST_v2")

dataset_path = natsorted([i for i in pyhistdir.rglob("*_densely_filtered_paths.csv") if "LungAOEC" in str(i)])

number_patches = 0
path_patches = {}
for wsi_patches in tqdm(dataset_path, desc="Selecting patches to extract features"):

    csv_instances = pd.read_csv(wsi_patches).to_numpy()
    
    number_patches = number_patches + len(csv_instances)
    name = wsi_patches.parent.stem
    path_patches[name] = csv_instances

logging.info(f"Total number of patches for train/validation {number_patches}")

preprocess_moco = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg_moco.dataset.mean, std=cfg_moco.dataset.stddev),
        transforms.Resize(size=(model_moco.resize_param, model_moco.resize_param),
        antialias=True)
    ])

params_dataloader_moco = {'batch_size': 1,
                          'shuffle': False,
                          'pin_memory': True,
                          'num_workers': 2}

patches_dataset = Dataset_instance(path_patches, transform=None, preprocess=preprocess_moco)
generator = DataLoader(patches_dataset, **params_dataloader_moco)

encoder.eval()
feature_patches_dict = {}
with torch.no_grad():
    
    for i, (x_q, x_k) in enumerate(generator):

        x_q, x_k = x_q.to(device, non_blocking=True), x_k.to(device, non_blocking=True)

        q = encoder(x_q)
        q = q.squeeze().cpu().numpy()
        feature_patches_dict[path_patches.stem[i]] = q

# Loading csv with data split
k = 10

data_split = pd.read_csv(Path(datadir / f"{k}_fold_crossvalidation_data_split.csv"), index_col=0)

for fold, _ in data_split.iterrows():
    train_dataset = literal_eval(data_split.loc[fold]["images_train"])
    validation_dataset = literal_eval(data_split.loc[fold]["images_test"])
    train_labels = literal_eval(data_split.loc[fold]["labels_train"])
    validation_labels = literal_eval(data_split.loc[fold]["labels_test"])

# Test data

test_path = natsorted([i for i in pyhistdir.rglob("*_densely_filtered_paths.csv") if "Lung" in str(i) and "LungAOEC" not in str(i)])

path_patches_test = {}
for wsi_patches in tqdm(test_path, desc="Selecting patches to extract features"):

    csv_instances = pd.read_csv(wsi_patches).to_numpy()
    
    number_patches = number_patches + len(csv_instances)
    name = wsi_patches.parent.stem
    path_patches_test[name] = csv_instances

logging.info(f"Total number of patches for test{number_patches}")

# Load datasets
batch_size_bag = 1

# sampler = Balanced_Multimodal

# if (PREPROCESSED_DATA==True and AUGMENT_PROB_THRESHOLD>0.0):
params_train_bag = {'batch_size': batch_size_bag,
                    'shuffle': True}
# else:
# params_train_bag = {'batch_size': batch_size_bag,
#     'sampler': sampler(train_dataset,alpha=0.25)}
#     #'shuffle': True}

params_valid_bag = {'batch_size': 1,
		            'shuffle': False}

training_set_bag = Dataset_bag_MIL(train_dataset[:,0], train_dataset[:,1:])
training_generator_bag = DataLoader(training_set_bag, **params_train_bag)

validation_set_bag = Dataset_bag_MIL(validation_dataset[:,0], validation_dataset[:,1:])
validation_generator_bag = DataLoader(validation_set_bag, **params_valid_bag)

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



def evaluate_validation_set(net, generator):
    #accumulator for validation set
    y_pred = []
    y_true = []

    validation_loss = 0.0

    net.eval()


    with torch.no_grad():
        for wsi_id, labels in generator:

            label_wsi = labels[0].cpu().numpy().flatten()
	
            labels_local = labels.float().flatten().to(device, non_blocking=True)

            print("[" + str(i) + "/" + str(len(train_dataset)) + "], " + "inputs_bag: " + str(wsi_id))
            print("labels: " + str(label_wsi))


            filename_features = '/home/niccolo/ExamodePipeline/Colon_WSI_patches/magnifications/'+'magnification_10x/'+filename_wsi+'/'+filename_wsi+'_features_'+MoCo_TO_USE+'.npy'
            #filename_features = '/home/niccolo/ExamodePipeline/Colon_WSI_patches/magnifications/'+'magnification_10x/'+filename_wsi+'/'+filename_wsi+'_features.npy'

            with open(filename_features, 'rb') as f:
                features_np = np.load(f)
                f.close()

            #read csv with instances
            n_indices = len(features_np)
            #indices = np.random.choice(n_indices,n_indices,replace=False)
            indices = np.random.permutation(n_indices)
            features_np = features_np[indices]

            inputs = torch.as_tensor(features_np).float().to(device, non_blocking=True)

            logits_img, cls_img = net(None, inputs)
			
            #loss img
            loss_img = criterion(logits_img, labels_local)

            sigmoid_output_img = F.sigmoid(logits_img)
            outputs_wsi_np_img = sigmoid_output_img.cpu().data.numpy()
            validation_loss = validation_loss + ((1 / (i+1)) * (loss_img.item() - validation_loss))
			
            print("pred_img: " + str(outputs_wsi_np_img))
            print("validation_loss: " + str(validation_loss))

            output_norm = np.where(outputs_wsi_np_img > 0.5, 1, 0)

            y_pred = np.append(y_pred,output_norm)
            y_true = np.append(y_true,label_wsi)

            micro_accuracy_train = accuracy_micro(y_true, y_pred)
            print("micro_accuracy " + str(micro_accuracy_train))    
	
    return validation_loss

epoch = 0
iterations = int(len(train_dataset) / batch_size_bag)+1

tot_batches_training = iterations#int(len(train_dataset)/batch_size_bag)
best_loss = 100000.0

	#number of epochs without improvement
early_stop = cfg.training.early_stop
early_stop_cont = 0

batch_size = cfg.dataloader.batch_size


while (epoch<cfg.training.epochs and early_stop_cont<early_stop):


    validation_loss = 0.0

    #accumulator accuracy for the outputs
    is_best = False

    filenames_wsis = []
    pred_cancers = []
    pred_hgd = []
    pred_lgd = []
    pred_hyper = []
    pred_normal = []

    samples_fname = []
    positive_samples_fname = []
    negative_samples_fname = []
    hard_negative_samples_fname = []

    y_pred = []
    y_true = []

    dataloader_iterator = iter(training_generator_bag)

    net.train()

    if (flag_dataset=='finetune_pretrain'):
        for param in model.embedding.parameters():
            param.requires_grad = False

        for param in model.attention.parameters():
            param.requires_grad = False
			
        for param in model.embedding_before_fc.parameters():
            param.requires_grad = False		

    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters CNN.')

    for i in range(iterations):
        print('[%d], %d / %d ' % (epoch, i, tot_batches_training))
        try:
            ID, labels = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(training_generator_bag)
            ID, labels = next(dataloader_iterator)
            #inputs: bags, labels: labels of the bags
			
        labels_np = labels.cpu().data.numpy().flatten()

        ID = ID[0]
        #instances_filename_sample = generate_list_instances(ID)
        diagnosis_wsi = get_diagnosis(ID)

        #embedding
        filename_wsi = ID

        labels_local = labels.float().flatten().to(device, non_blocking=True)

        print("[" + str(i) + "/" + str(len(train_dataset)) + "], " + "inputs_bag: " + str(filename_wsi))
        print("labels: " + str(labels_np))
	

        prob_pre = np.random.rand(1)[0]

        #pipeline_transform = generate_transformer(labels_np[i_wsi])
        pipeline_transform = generate_transformer(labels_np)
        instances_filename_sample = generate_list_instances(ID)

        csv_instances = pd.read_csv(instances_filename_sample, sep=',', header=None).values
        n_elems = len(csv_instances)
        #embedding

        n_elems = len(csv_instances)
	
        num_workers = cfg.dataloader.num_workers

        if (n_elems > batch_size_instance):
            pin_memory = True
        else:
            pin_memory = False

        params_instance = {'batch_size': batch_size_instance,
                'num_workers': num_workers,
                'pin_memory': pin_memory,
                'shuffle': True}

        instances = Dataset_instance_MIL(csv_instances,mode,pipeline_transform)
        generator = DataLoader(instances, **params_instance)

		net.eval()

        features = []
        with torch.no_grad():
            for instances in training_generator_instance:
                instances = instances.to(device, non_blocking=True)

                # forward + backward + optimize
                feats = net.conv_layers(instances)
                feats = feats.view(-1, fc_input_features)
                feats_np = feats.cpu().data.numpy()

                features = np.append(features,feats_np)

				#del instances
			#del instances
        features_np = np.reshape(features,(n_elems,fc_input_features))

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
        validation_loss = validation_loss + ((1 / (i+1)) * (loss.item() - validation_loss))

        loss.backward() 

        optimizer.step()
        #scheduler.step()

        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        #bert_model.zero_grad()

        print()
        print("pred_img: " + str(outputs_wsi_np_img))
        print("validation_loss: " + str(validation_loss))

        filenames_wsis = np.append(filenames_wsis, filename_wsi)
        pred_cancers = np.append(pred_cancers, outputs_wsi_np_img[0])
        pred_hgd = np.append(pred_hgd, outputs_wsi_np_img[1])
        pred_lgd = np.append(pred_lgd, outputs_wsi_np_img[2])
        pred_hyper = np.append(pred_hyper, outputs_wsi_np_img[3])
        pred_normal = np.append(pred_normal, outputs_wsi_np_img[4])

        output_norm = np.where(outputs_wsi_np_img > 0.5, 1, 0)

        y_pred = np.append(y_pred,output_norm)
        y_true = np.append(y_true,labels_np)

        micro_accuracy_train = accuracy_micro(y_true, y_pred)
        print("micro_accuracy " + str(micro_accuracy_train))    

    #save_training predictions
    filename_training_predictions = checkpoint_path+'training_predictions_'+str(epoch)+'.csv'

    File = {'filenames':filenames_wsis, 'pred_cancers':pred_cancers, 'pred_hgd':pred_hgd,'pred_lgd':pred_lgd, 'pred_hyper':pred_hyper, 'pred_normal':pred_normal}

    df = pd.DataFrame(File,columns=['filenames','pred_cancers','pred_hgd','pred_lgd','pred_hyper','pred_normal'])
    np.savetxt(filename_training_predictions, df.values, fmt='%s',delimiter=',')

    #bert_model.eval()

    samples_fname = []
    positive_samples_fname = []
    negative_samples_fname = []
    hard_negative_samples_fname = []

    print("evaluating validation")
    valid_loss = evaluate_validation_set(net, epoch, validation_generator_bag)

    #save validation
    filename_val = validation_checkpoints+'validation_value_'+str(epoch)+'.csv'
    array_val = [valid_loss]
    File = {'val':array_val}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(filename_val, df.values, fmt='%s',delimiter=',')

    #save_hyperparameters
    filename_hyperparameters = checkpoint_path+'hyperparameters.csv'
    array_n_classes = model.num_classes
    array_lr = [str(lr)]
    array_embedding = [EMBEDDING_bool]
    File = {'n_classes':array_n_classes, 'lr':array_lr, 'embedding':array_embedding}

    df = pd.DataFrame(File,columns=['n_classes','lr','embedding','modality'])
    np.savetxt(filename_hyperparameters, df.values, fmt='%s',delimiter=',')

    if (epoch>=THRESHOLD_CLASSIFICATION):
        if (best_loss>valid_loss):
            early_stop_cont = 0
            print ("=> Saving a new best model")
            print("previous loss : " + str(best_loss) + ", new loss function: " + str(valid_loss))
            best_loss = valid_loss

            try:
                torch.save(net.state_dict(), model_weights_filename_CNN,_use_new_zipfile_serialization=False)
            except:
                try:
                    torch.save(net.state_dict(), model_weights_filename_CNN)    
                except:
                    torch.save(net, model_weights_filename_CNN)

            filename_training_predictions = checkpoint_path+'training_predictions_best.csv'

            File = {'filenames':filenames_wsis, 'pred_cancers':pred_cancers, 'pred_hgd':pred_hgd,'pred_lgd':pred_lgd, 'pred_hyper':pred_hyper, 'pred_normal':pred_normal}

            df = pd.DataFrame(File,columns=['filenames','pred_cancers','pred_hgd','pred_lgd','pred_hyper','pred_normal'])
            np.savetxt(filename_training_predictions, df.values, fmt='%s',delimiter=',')

        else:
            early_stop_cont = early_stop_cont+1
		

    #save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, model_weights_filename_checkpoint)

    epoch = epoch+1
    if (early_stop_cont == early_stop):
        print("EARLY STOPPING")

torch.cuda.empty_cache()