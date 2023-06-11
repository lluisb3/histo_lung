#!/bin/bash 

python3 -m preprocessing.store_features --config_file config_Features --exp_name_moco MoCo_resnet34_v2
python3 -m preprocessing.store_features --config_file config_Features2 --exp_name_moco MoCo_resnet34_v2