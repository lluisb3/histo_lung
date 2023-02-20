from pathlib import Path
import yaml
import torch
import numpy as np
import albumentations as A
from training import model_option

thispath = Path(__file__).resolve()

seed = 33
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


def main():
    # read the configuration file
    print(f"Device: {device}")

    config_path = str(thispath.parent / 'config.yml')
    print(f"With configuration file in: {config_path}")
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    # use the configuration for the network
    model_arguments = cfg['model']



    outputdir = Path(thispath.parent.parent / "trained_models" / f"{cfg['name_experiment']}")
    Path(outputdir).mkdir(exist_ok=True, parents=True)

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
    DataAugmentation = transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(70),
                                                                   transforms.RandomVerticalFlip(),
                                                                   transforms.RandomHorizontalFlip(),
                                                                   transforms.RandomAffine(degrees=0, scale=(.9, 1.1),
                                                                                           translate=(0.2, 0.2),
                                                                                           shear=30),
                                                                   transforms.GaussianBlur([3, 3])]),
                                              p=cfg['data_aug']['prob'])

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          DataAugmentation,
                                          transforms.Resize(size=(resize_param, resize_param))])
    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(resize_param, resize_param))])

    # use the configuration for the dataset
    dataset_arguments = cfg['dataset']
    dataset_train = SkinLesionDataset(challenge_name=dataset_arguments['challenge_name'],
                                      dataset_set='train',
                                      dataset_mean=dataset_arguments['mean'],
                                      dataset_std=dataset_arguments['stddev'],
                                      transform=transform_train,
                                      seg_image=dataset_arguments['use_masks'])
    dataset_val = SkinLesionDataset(challenge_name=dataset_arguments['challenge_name'],
                                    dataset_set='val',
                                    dataset_mean=dataset_arguments['mean'],
                                    dataset_std=dataset_arguments['stddev'],
                                    transform=transform_val,
                                    seg_image=dataset_arguments['use_masks'])
    datasets = {'train': dataset_train, 'val': dataset_val}
    # use the configuration for the dataloader
    dataset_arguments = cfg['dataloaders']
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=dataset_arguments['train_batch_size'],
                                  shuffle=True,
                                  num_workers=dataset_arguments['num_workers'],
                                  pin_memory=True)
    dataloader_valid = DataLoader(dataset_val,
                                  batch_size=dataset_arguments['val_batch_size'],
                                  num_workers=dataset_arguments['num_workers'],
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
