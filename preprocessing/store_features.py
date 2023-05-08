from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils import data
import torch.utils.data as dat
from torchvision import transforms
import numpy as np
import pandas as pd
from training.mil import MIL_model
from training.models import ModelOption
from training.utils_trainig import yaml_load, edict2dict, get_generator_instances
import logging
import yaml
import click
from tqdm import tqdm
from natsort import natsorted

thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.option(
    "--config_file",
    default="config_Features",
    prompt="Name of the config file without extension",
    help="Name of the config file without extension",
)
@click.option(
    "--exp_name_moco",
    default="MoCo_convnext",
    prompt="Name of the MoCo experiment",
    help="Name of the MoCo experiment",
)
def main(config_file, exp_name_moco):
	# Read the configuration file
	configdir = Path(thispath.parent / f"{config_file}.yml")
	cfg = yaml_load(configdir)

	# Create directory to save the resuls
	outputdir = Path(datadir / "Saved_features" / cfg.experiment_name)
	Path(outputdir).mkdir(exist_ok=True, parents=True)

	# Save config parameters for experiment
	with open(Path(f"{outputdir}/config_{cfg.experiment_name}.yml"), 'w') as yaml_file:
		yaml.dump(edict2dict(cfg), yaml_file, default_flow_style=False)

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


	preprocess = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev),
			transforms.Resize(size=(model.resize_param, model.resize_param),
			antialias=True)
		])


	hidden_space_len = cfg.model.hidden_space_len

	net = MIL_model(model, hidden_space_len, cfg)
	net.load_state_dict(checkpoint_moco["encoder_state_dict"], strict=False)
	net.to(device)
	net.eval()

	pyhistdir = Path(datadir / "Mask_PyHIST_v2") 

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


	for wsi_id, path_for_patches in tqdm(patches_path.items()):

		n_elems = len(path_for_patches)

		training_generator_instance = get_generator_instances(patches_path[wsi_id], 
                                                              preprocess,
                                                              cfg.dataloader.batch_size, 
                                                              None,
                                                              cfg.dataloader.num_workers)

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
		features_np = np.reshape(features, (n_elems, net.fc_input_features))

		np.save(outputdir / f"{wsi_id}.npy", features_np)


if __name__ == '__main__':
    main()
