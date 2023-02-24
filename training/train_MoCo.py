from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A
from torchvision import transforms
from training import model_option


thispath = Path(__file__).resolve()

seed = 33
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


def main():
    
    print(f"CUDA current device {torch.cuda.current_device()}")
    print(f"CUDA devices available {torch.cuda.device_count()}")

    # read the configuration file
    config_path = str(thispath.parent / 'config.yml')
    print(f"With configuration file in: {config_path}")
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    # use the configuration for the network
    model_arguments = cfg['model']


    # Create directories for the outputs
    outputdir = Path(thispath.parent.parent / "trained_models" / f"{cfg['name_experiment']}")
    Path(outputdir).mkdir(exist_ok=True, parents=True)
    #type of MIL
    Path(outputdir / "MoCo").mkdir(exist_ok=True, parents=True)
    #network used
    Path(outputdir / cfg['model']['model_name']).mkdir(exist_ok=True, parents=True)
    #checkpoint dir where put csv with loss function, metrics evaluation, setup
    Path(outputdir / "checkpoints_MIL").mkdir(exist_ok=True, parents=True)

    model_weights_filename = outputdir +'MoCo.pt'
    model_weights_temporary_filename = outputdir +'MoCo_temporary.pt'

    net, resize_param = model_option(model_arguments['model_name'],
                                     model_arguments['num_classes'],
                                     freeze=model_arguments['freeze_weights'],
                                     num_freezed_layers=model_arguments['num_frozen_layers'],
                                     seg_mask=cfg['dataset']['use_masks'],
                                     dropout=model_arguments['dropout']
                                     )
    # Data transformations
    #DATA AUGMENTATION
    
    prob = 0.75
    prob_config =  p=cfg['data_aug']['prob']


    pipeline_transform_paper = A.Compose([
        #A.RandomScale(scale_limit=(-0.005,0.005), interpolation=2, p=prob),
        #A.RandomCrop(height=220, width=220, p=prob),
        #A.Resize(224,224,always_apply=True),
        #A.MotionBlur(blur_limit=3, p=prob),
        #A.MedianBlur(blur_limit=3, p=prob),
        #A.CropAndPad(percent=(-0.01, -0.05),pad_mode=1,always_apply=True),
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1), always_apply=True),
        A.VerticalFlip(p=prob),
        A.HorizontalFlip(p=prob),
        A.RandomRotate90(p=prob),
        #A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-20,10),val_shift_limit=(-8,8),always_apply=True),
        A.ColorJitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, always_apply=True),
        A.GaussianBlur (blur_limit=(1, 3), sigma_limit=0, always_apply=True),
        #A.HueSaturationValue(hue_shift_limit=(-25,10),sat_shift_limit=(-25,15),val_shift_limit=(-15,15),always_apply=True),
        #A.RGBShift (r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=True, p=prob),
        #A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=prob),
        #A.RandomBrightness(limit=0.2, p=prob),
        #A.RandomContrast(limit=0.2, p=prob),
        #A.GaussNoise(p=prob),
        #A.ElasticTransform(alpha=2,border_mode=4, sigma=20, alpha_affine=20, p=prob, always_apply=True),
        #A.GridDistortion(num_steps=2, distort_limit=0.2, interpolation=1, border_mode=4, p=prob),
        #A.GlassBlur(sigma=0.3, max_delta=2, iterations=1, p=prob),
        #A.OpticalDistortion (distort_limit=0.2, shift_limit=0.2, interpolation=1, border_mode=4, value=None, p=prob),
        #A.GridDropout (ratio=0.3, unit_size_min=3, unit_size_max=40, holes_number_x=3, holes_number_y=3, shift_x=1, shift_y=10, random_offset=True, fill_value=0, p=prob),
        #A.Equalize(p=prob),
        #A.Posterize(p=prob, always_apply=True),
        #A.RandomGamma(p=prob, always_apply=True),
        #A.Superpixels(p_replace=0.05, n_segments=100, max_size=128, interpolation=1, p=prob),
        #A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3, p=prob),
        A.ToGray(p=0.2),
        #A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=0, p=prob),
        #A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=255, p=prob),
        ])
    

    pipeline_transform = A.Compose([
        #A.RandomScale(scale_limit=(-0.005,0.005), interpolation=2, p=prob),
        #A.RandomCrop(height=220, width=220, p=prob),
        #A.Resize(224,224,always_apply=True),
        #A.MotionBlur(blur_limit=3, p=prob),
        #A.MedianBlur(blur_limit=3, p=prob),
        #A.CropAndPad(percent=(-0.01, -0.05),pad_mode=1,always_apply=True),
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1), p = prob_config),
        A.VerticalFlip(p=prob_config),
        A.HorizontalFlip(p=prob_config),
        A.RandomRotate90(p=prob_config),
        A.HueSaturationValue(hue_shift_limit=(-25,15),sat_shift_limit=(-20,30),val_shift_limit=(-15,15),always_apply=True),
        A.ColorJitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=prob_config),
        #A.GaussianBlur (blur_limit=(1, 3), sigma_limit=0, p=prob),
        #A.HueSaturationValue(hue_shift_limit=(-25,10),sat_shift_limit=(-25,15),val_shift_limit=(-15,15),always_apply=True),
        #A.RGBShift (r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=True, p=prob),
        #A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=prob),
        #A.RandomBrightness(limit=0.2, p=prob),
        #A.RandomContrast(limit=0.2, p=prob),
        #A.GaussNoise(p=prob),
        A.ElasticTransform(alpha=200, sigma=10, alpha_affine=10, interpolation=2, border_mode=4, p=prob_config),
        A.GridDistortion(num_steps=1, distort_limit=0.2, interpolation=1, border_mode=4, p=prob_config),
        A.GlassBlur(sigma=0.1, max_delta=1, iterations=1, p=prob_config),
        A.OpticalDistortion (distort_limit=0.2, shift_limit=0.2, interpolation=1, border_mode=4, value=None, p=prob_config),
        #A.GridDropout (ratio=0.3, unit_size_min=3, unit_size_max=40, holes_number_x=3, holes_number_y=3, shift_x=1, shift_y=10, random_offset=True, fill_value=0, p=prob),
        A.Equalize(p=prob_config),
        #A.Posterize(p=prob, always_apply=True),
        #A.RandomGamma(p=prob, always_apply=True),
        #A.Superpixels(p_replace=0.05, n_segments=100, max_size=128, interpolation=1, p=prob),
        #A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3, p=prob),
        A.ToGray(p=0.2),
        #A.Affine(shear = (-5, 5), translate_px = (-5,5), p = prob),
        #A.Affine(translate_px = (-5,5), p = 1),
        #A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=0, p=prob),
        #A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=255, p=prob),
        ])

    
    pipeline_transform_soft = A.Compose([
        #A.ElasticTransform(alpha=0.01,p=p_soft),
        #A.RGBShift (r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=True, p=p_soft),
        A.HueSaturationValue(hue_shift_limit=(-10,10),sat_shift_limit=(-10,10),val_shift_limit=(-5,5),p=prob_config),
        A.VerticalFlip(p=prob_config),
        A.HorizontalFlip(p=prob_config),
        A.RandomRotate90(p=prob_config),
        #A.HueSaturationValue(hue_shift_limit=(-25,10),sat_shift_limit=(-25,15),val_shift_limit=(-15,15),p=p_soft),
        #A.CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=p_soft),
        #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=p_soft),
        #A.RandomBrightness(limit=0.1, p=p_soft),
        #A.RandomContrast(limit=0.1, p=p_soft),
        ])
        
    #DATA NORMALIZATION
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(resize_param, resize_param))
    ])
                                             

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          DataAugmentation,
                                          transforms.Resize(size=(resize_param, resize_param))])
    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(resize_param, resize_param))])

    # use the configuration for the dataset
    dataset_arguments = cfg['dataset']
    dataset_train = SkinLesionDataset(dataset_set='train',
                                      dataset_mean=dataset_arguments['mean'],
                                      dataset_std=dataset_arguments['stddev'],
                                      transform=transform_train,
                                      seg_image=dataset_arguments['use_masks'])
    dataset_val = SkinLesionDataset(dataset_set='val',
                                    dataset_mean=dataset_arguments['mean'],
                                    dataset_std=dataset_arguments['stddev'],
                                    transform=transform_val,
                                    seg_image=dataset_arguments['use_masks'])
    
    datasets = {'train': dataset_train, 'val': dataset_val}

    # use the configuration for the dataloader
    dataloader_arguments = cfg['dataloaders']
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=dataloader_arguments['batch_size'],
                                  shuffle=True,
                                  num_workers=dataloader_arguments['num_workers'],
                                  pin_memory=True)
    dataloader_valid = DataLoader(dataset_val,
                                  batch_size=dataloader_arguments['batch_size'],
                                  num_workers=dataloader_arguments['num_workers'],
                                  pin_memory=True)
    dataloaders = {'train': dataloader_train, 'val': dataloader_valid}

    # loss function
    if 'criterion_args' in cfg['training']:
        if cfg['training']['criterion_args'].get('weight') is not None:
            holder = cfg['training']['criterion_args']['weight'].copy()
            cfg['training']['criterion_args']['weight'] = torch.tensor(cfg['training']['criterion_args']['weight'],
                                                                       dtype=torch.float,
                                                                       device=device)
        criterion = getattr(torch.nn, cfg['training']['criterion'])(**cfg['training']['criterion_args'])
    else:
        criterion = getattr(torch.nn, cfg['training']['criterion'])()

    # Optimizer
    optimizer = getattr(torch.optim, cfg['training']['optimizer'])
    optimizer = optimizer(net.parameters(), **cfg['training']['optimizer_args'])

    scheduler = getattr(torch.optim.lr_scheduler, cfg['training']['lr_scheduler'])
    scheduler = scheduler(optimizer, **cfg['training']['lr_scheduler_args'])
    # **d means "treat the key-value pairs in the dictionary as additional named arguments to this function call."
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    train(net, datasets, dataloaders, criterion, optimizer, scheduler, cfg)


if __name__ == '__main__':
    main()
