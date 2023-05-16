#!/bin/bash 

# python3 -m training.train_MIL --config_file config_MIL --exp_name_moco MoCo_resnet101_scheduler_5
# python3 -m training.train_MIL --config_file config_MIL2 --exp_name_moco MoCo_convnext
# python3 -m training.train_MIL --config_file config_MIL3 --exp_name_moco MoCo_convnext
# python3 -m training.train_MIL --config_file config_MIL4 --exp_name_moco MoCo_convnext
# python3 -m training.train_MIL --config_file config_MIL5 --exp_name_moco MoCo_convnext
# python3 -m training.train_MIL --config_file config_MIL6 --exp_name_moco MoCo_resnet101_scheduler_5
# python3 -m training.train_MIL --config_file config_MIL7 --exp_name_moco MoCo_convnext
python3 -m training.train_MIL --config_file config_MIL8 --exp_name_moco MoCo_convnext_40_v2
python3 -m training.train_MIL --config_file config_MIL9 --exp_name_moco MoCo_convnext_40_v2
python3 -m training.train_MIL --config_file config_MIL10 --exp_name_moco MoCo_convnext_40_v2
python3 -m training.train_MIL --config_file config_MIL11 --exp_name_moco MoCo_convnext_40_v2
# python3 -m training.train_MIL --config_file config_MIL12 --exp_name_moco MoCo_convnext_40_v2

# python3 -m preprocessing.store_features --config_file config_Features --exp_name_moco MoCo_convnext_40_v2
# python3 -m preprocessing.store_features --config_file config_Features2 --exp_name_moco MoCo_convnext_40_v2