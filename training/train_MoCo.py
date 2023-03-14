from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import albumentations as A
from torchvision import transforms
from training import ModelOption, Encoder, generate_list_instances, momentum_step, contrastive_loss

thispath = Path(__file__).resolve()


def train(train_dataset, cfg, econcer, momentum_encoder):
    # Training
    print('\nStart training!')
    epoch = 0 

    losses_train = []
    fnames_patches = []

    new_patches = 0

    for idx_wsi, wsi in tqdm(enumerate(train_dataset), desc="Selecting all patches for training"):

        fname = wsi[0]
    
        print(idx_wsi, fname)

        csv_fname = generate_list_instances(fname)
        csv_instances = pd.read_csv(csv_fname, sep=',', header=None).values
        l_csv = len(csv_instances)

        #new_patches = new_patches + len(csv_instances)
        new_patches = new_patches + l_csv
        fnames_patches = np.append(fnames_patches, csv_instances)

        iterations_per_epoch = int(len(fnames_patches) / batch_size) + 1


    training_arguments = cfg["training"]

    moco_m = training_arguments["moco_m"]
    temperature = training_arguments["temperature"]

    batch_size = cfg["dataloaders"]["batch_size"]
    #number of epochs without improvement
    early_stop = training_arguments["early_stop"]
    early_stop_cont = 0
    # validation_checkpoints = checkpoint_path+'validation_losses/'
    # create_dir(validation_checkpoints)
        #number of epochs without improvement
    iterations = int(len(train_dataset) / batch_size_bag)#+100
    #iterations = 600

    tot_batches_training = iterations#int(len(train_dataset)/batch_size_bag)
    best_loss = 100000.0

    tot_iterations = training_arguments["epochs"] * iterations_per_epoch
    cont_iterations_tot = 0

    while (epoch < training_arguments["epochs"] and early_stop_cont < early_stop):
        total_iters = 0 
    
        #accumulator loss for the outputs
        train_loss = 0.0
        train_loss_domain = 0.0
        train_loss_moco = 0.0

        #if loss function lower
        is_best = False

        print('\n[3 / 3]. Initializing a queue with %d keys.' % training_arguments["num_keys"])
        queue = []

        dataloader_iterator = iter(training_generator_bag)
    
        new_patches = 0

        num_workers = 16
        params_instance = {'batch_size': batch_size,
                           'shuffle': True,
                           'pin_memory': True,
                           'drop_last':True,
                           'num_workers': num_workers}

        mode = 'valid'
        instances = Dataset_instance(fnames_patches, mode)
        generator = DataLoader(instances, **params_instance)

        with torch.no_grad():
            for i, (_, img) in enumerate(generator):
                key_feature = momentum_encoder(img.to(device, non_blocking=True))
                key_feature = torch.nn.functional.normalize(key_feature, dim=1)
                queue.append(key_feature)

                if i == (training_arguments["num_keys"] / batch_size) - 1:
                    break
            queue = torch.cat(queue, dim=0)

        print("queue done")

        dataloader_iterator = iter(training_generator_bag)
    
        new_patches = 0

        num_workers = 16
        params_instance = {'batch_size': batch_size,
                           'shuffle': True,
                           'pin_memory': True,
                           'drop_last':True,
                           'num_workers': num_workers}

        mode = 'train'
        instances = Dataset_instance(fnames_patches, mode)
        generator = DataLoader(instances, **params_instance)

        dataloader_iterator = iter(training_generator_bag)

        j = 0

        encoder.train()
        momentum_encoder.train()

        for a, (x_q, x_k) in enumerate(generator):
        
            p = float(cont_iterations_tot + epoch * tot_iterations) / training_arguments["epochs"] / tot_iterations

            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # Preprocess
            #momentum_encoder.train()
            #momentum_encoder.zero_grad()
            #encoder.train()
            #encoder.zero_grad()

            # Shffled BN : shuffle x_k before distributing it among GPUs (Section. 3.3)
            if shuffle_bn:
                idx = torch.randperm(x_k.size(0))
                x_k = x_k[idx]

            # x_q, x_k : (N, 3, 64, 64)            
            x_q, x_k = x_q.to(device, non_blocking=True), x_k.to(device, non_blocking=True)

            q = encoder(x_q) # q : (N, 128)

            with torch.no_grad():
                k = momentum_encoder(x_k).detach() # k : (N, 128)
    
            #q = torch.div(q,torch.norm(q,dim=1).reshape(-1,1))
            #k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1))	
    
            q = torch.nn.functional.normalize(q, dim=1)
            k = torch.nn.functional.normalize(k, dim=1)

            #q = torch.nn.functional.normalize(q, dim=0)
            #k = torch.nn.functional.normalize(k, dim=0)

            # Shuffled BN : unshuffle k (Section. 3.3)
            if shuffle_bn:
                k_temp = torch.zeros_like(k)
                for a, j in enumerate(idx):
                    k_temp[j] = k[a]
            k = k_temp
            """
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, queue.t()]        

            # Positive sampling q & k
            #l_pos = torch.sum(q * k, dim=1, keepdim=True) # (N, 1)
            #print("l_pos",l_pos)

            # Negative sampling q & queue
            #l_neg = torch.mm(q, queue.t()) # (N, 4096)
            #print("l_neg",l_neg)

            # Logit and label
            logits = torch.cat([l_pos, l_neg], dim=1) / temperature # (N, 4097) witi label [0, 0, ..., 0]
            labels = torch.zeros(logits.size(0), dtype=torch.long).to(device)

            # Get loss and backprop
            loss_moco = criterion(logits, labels)
            """
            loss_moco = contrastive_loss(q, k, queue)

            loss = loss_moco #+ loss_domains

            loss.backward()

            # Encoder update
            optimizer.step()

            momentum_encoder.zero_grad(set_to_none=True)
            encoder.zero_grad(set_to_none=True)

            # Momentum encoder update
            momentum_step(m=moco_m)

            # Update dictionary
            #queue = torch.cat([k, queue[:queue.size(0) - k.size(0)]], dim=0)
            queue = update_queue(queue, k)
            #print(queue.shape)

            # Print a training status, save a loss value, and plot a loss graph.

            train_loss_moco = train_loss_moco + ((1 / (total_iters+1)) * (loss_moco.item() - train_loss_moco)) 
            total_iters = total_iters + 1
            cont_iterations_tot = cont_iterations_tot + 1
            train_loss = train_loss_moco + train_loss_domain

            print('[Epoch : %d / Total iters : %d] : loss_moco :%f, loss_domain :%f ...' %(epoch, total_iters, train_loss_moco, train_loss_domain))
    
            if (total_iters%100==True):
                print('a')
                if (best_loss>train_loss_moco):
                    early_stop_cont = 0
                    print ("=> Saving a new best model")
                    print("previous loss : " + str(best_loss) + ", new loss function: " + str(train_loss_moco))
                    best_loss = train_loss_moco
                    try:
                        torch.save(encoder.state_dict(), model_weights_filename,_use_new_zipfile_serialization=False)
                    except:
                        torch.save(encoder.state_dict(), model_weights_filename)
                else:

                    try:
                        torch.save(encoder.state_dict(), model_weights_temporary_filename,_use_new_zipfile_serialization=False)
                    except:
                        torch.save(encoder.state_dict(), model_weights_temporary_filename)

                torch.cuda.empty_cache()
    
            # Update learning rate
        #update_lr(epoch)

        print("epoch "+str(epoch)+ " train loss: " + str(train_loss))

        print("evaluating validation")
        """
        valid_loss = validate(epoch, validation_generator_bag)

        #save validation
        filename_val = validation_checkpoints+'validation_value_'+str(epoch)+'.csv'
        array_val = [valid_loss]
        File = {'val':array_val}
        df = pd.DataFrame(File,columns=['val'])
        np.savetxt(filename_val, df.values, fmt='%s',delimiter=',')

        #save_hyperparameters
        filename_hyperparameters = checkpoint_path+'hyperparameters.csv'
        array_lr = [str(lr)]
        array_opt = [optimizer_str]
        array_wt_decay = [str(weight_decay)]
        array_embedding = [EMBEDDING_bool]
        File = {'opt':array_opt, 'lr':array_lr,'wt_decay':array_wt_decay,'array_embedding':EMBEDDING_bool}

        df = pd.DataFrame(File,columns=['opt','lr','wt_decay','array_embedding'])
        np.savetxt(filename_hyperparameters, df.values, fmt='%s',delimiter=',')
        """

        epoch = epoch+1
        if (early_stop_cont == early_stop):
            print("EARLY STOPPING")


def main():

    datadir = Path(thispath.parent.parent / "data")

    print(f"CUDA current device {torch.cuda.current_device()}")
    print(f"CUDA devices available {torch.cuda.device_count()}")

    # read the configuration file
    config_path = str(thispath.parent / 'config.yml')
    print(f"With configuration file in: {config_path}")
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # Create directories for the outputs
    outputdir = Path(thispath.parent.parent / "trained_models" / f"{cfg['experiment_name']}")
    Path(outputdir).mkdir(exist_ok=True, parents=True)

    # type of MIL
    Path(outputdir / "MoCo").mkdir(exist_ok=True, parents=True)

    # network used
    Path(outputdir / cfg['model']['model_name']).mkdir(exist_ok=True, parents=True)

    # checkpoint dir where put csv with loss function, metrics evaluation, setup
    Path(outputdir / "checkpoints_MIL").mkdir(exist_ok=True, parents=True)

    seed = 33
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Load CSV k-fold-crossvalidation
    from ast import literal_eval

    k = cfg['dataset']['k']
    fold = 0

    data_split = pd.read_csv(Path(datadir / f"{k}_fold_crossvalidation_data_split.csv"), index_col=0)

    # Data fold 0
    train_data = literal_eval(data_split.loc[fold]["images_train"])
    test_data = literal_eval(data_split.loc[fold]["images_test"])
    train_labels = literal_eval(data_split.loc[fold]["labels_train"])
    test_labels = literal_eval(data_split.loc[fold]["labels_test"])

    # use the configuration for the network
    model_arguments = cfg['model']

    # Load pretrained model
    model = ModelOption(model_arguments['model_name'],
                model_arguments['num_classes'],
                freeze=model_arguments['freeze_weights'],
                num_freezed_layers=model_arguments['num_frozen_layers'],
                seg_mask=cfg['dataset']['use_masks'],
                dropout=model_arguments['dropout']
                )    

    training_arguments = cfg["training"]

    moco_dim = training_arguments["moco_dim"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(model, model_arguments, moco_dim).to(device)
    momentum_encoder = Encoder(model, model_arguments, dim=moco_dim).to(device)
    
    # Data transformations
    # DATA AUGMENTATION
    
    prob_augmentation = cfg['data_aug']['prob']

    # pipeline_transform_paper = A.Compose([
    #     # A.RandomScale(scale_limit=(-0.005,0.005), interpolation=2, p=prob),
    #     # A.RandomCrop(height=220, width=220, p=prob),
    #     # A.Resize(224,224,always_apply=True),
    #     # A.MotionBlur(blur_limit=3, p=prob),
    #     # A.MedianBlur(blur_limit=3, p=prob),
    #     # A.CropAndPad(percent=(-0.01, -0.05),pad_mode=1,always_apply=True),
    #     A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1), always_apply=True),
    #     A.VerticalFlip(p=prob),
    #     A.HorizontalFlip(p=prob),
    #     A.RandomRotate90(p=prob),
    #     # A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-20,10),val_shift_limit=(-8,8),always_apply=True),
    #     A.ColorJitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, always_apply=True),
    #     A.GaussianBlur (blur_limit=(1, 3), sigma_limit=0, always_apply=True),
    #     # A.HueSaturationValue(hue_shift_limit=(-25,10),sat_shift_limit=(-25,15),val_shift_limit=(-15,15),always_apply=True),
    #     # A.RGBShift (r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=True, p=prob),
    #     # A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=prob),
    #     # A.RandomBrightness(limit=0.2, p=prob),
    #     # A.RandomContrast(limit=0.2, p=prob),
    #     # A.GaussNoise(p=prob),
    #     # A.ElasticTransform(alpha=2,border_mode=4, sigma=20, alpha_affine=20, p=prob, always_apply=True),
    #     # A.GridDistortion(num_steps=2, distort_limit=0.2, interpolation=1, border_mode=4, p=prob),
    #     # A.GlassBlur(sigma=0.3, max_delta=2, iterations=1, p=prob),
    #     # A.OpticalDistortion (distort_limit=0.2, shift_limit=0.2, interpolation=1, border_mode=4, value=None, p=prob),
    #     # A.GridDropout (ratio=0.3, unit_size_min=3, unit_size_max=40, holes_number_x=3, holes_number_y=3, shift_x=1, shift_y=10, random_offset=True, fill_value=0, p=prob),
    #     # A.Equalize(p=prob),
    #     # A.Posterize(p=prob, always_apply=True),
    #     # A.RandomGamma(p=prob, always_apply=True),
    #     # A.Superpixels(p_replace=0.05, n_segments=100, max_size=128, interpolation=1, p=prob),
    #     # A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3, p=prob),
    #     A.ToGray(p=0.2),
    #     # A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=0, p=prob),
    #     # A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=255, p=prob),
    #     ])
    

    pipeline_transform = A.Compose([
        # A.RandomScale(scale_limit=(-0.005,0.005), interpolation=2, p=prob),
        # A.RandomCrop(height=220, width=220, p=prob),
        # A.Resize(224,224,always_apply=True),
        # A.MotionBlur(blur_limit=3, p=prob),
        # A.MedianBlur(blur_limit=3, p=prob),
        # A.CropAndPad(percent=(-0.01, -0.05),pad_mode=1,always_apply=True),
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1), p = prob_augmentation),
        A.VerticalFlip(p=prob_augmentation),
        A.HorizontalFlip(p=prob_augmentation),
        A.RandomRotate90(p=prob_augmentation),
        A.HueSaturationValue(hue_shift_limit=(-25,15),sat_shift_limit=(-20,30),val_shift_limit=(-15,15),always_apply=True),
        A.ColorJitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=prob_augmentation),
        # A.GaussianBlur (blur_limit=(1, 3), sigma_limit=0, p=prob),
        # A.HueSaturationValue(hue_shift_limit=(-25,10),sat_shift_limit=(-25,15),val_shift_limit=(-15,15),always_apply=True),
        # A.RGBShift (r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=True, p=prob),
        # A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=prob),
        # A.RandomBrightness(limit=0.2, p=prob),
        # A.RandomContrast(limit=0.2, p=prob),
        # A.GaussNoise(p=prob),
        A.ElasticTransform(alpha=200, sigma=10, alpha_affine=10, interpolation=2, border_mode=4, p=prob_augmentation),
        A.GridDistortion(num_steps=1, distort_limit=0.2, interpolation=1, border_mode=4, p=prob_augmentation),
        A.GlassBlur(sigma=0.1, max_delta=1, iterations=1, p=prob_augmentation),
        A.OpticalDistortion (distort_limit=0.2, shift_limit=0.2, interpolation=1, border_mode=4, value=None, p=prob_augmentation),
        # A.GridDropout (ratio=0.3, unit_size_min=3, unit_size_max=40, holes_number_x=3, holes_number_y=3, shift_x=1, shift_y=10, random_offset=True, fill_value=0, p=prob),
        A.Equalize(p=prob_augmentation),
        # A.Posterize(p=prob, always_apply=True),
        # A.RandomGamma(p=prob, always_apply=True),
        # A.Superpixels(p_replace=0.05, n_segments=100, max_size=128, interpolation=1, p=prob),
        # A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3, p=prob),
        A.ToGray(p=0.2),
        # A.Affine(shear = (-5, 5), translate_px = (-5,5), p = prob),
        # A.Affine(translate_px = (-5,5), p = 1),
        # A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=0, p=prob),
        # A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=255, p=prob),
        ])

    # pipeline_transform_soft = A.Compose([
    #     # A.ElasticTransform(alpha=0.01,p=p_soft),
    #     # A.RGBShift (r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=True, p=p_soft),
    #     A.HueSaturationValue(hue_shift_limit=(-10,10),sat_shift_limit=(-10,10),val_shift_limit=(-5,5),p=prob_augmentation),
    #     A.VerticalFlip(p=prob_augmentation),
    #     A.HorizontalFlip(p=prob_augmentation),
    #     A.RandomRotate90(p=prob_augmentation),
    #     # A.HueSaturationValue(hue_shift_limit=(-25,10),sat_shift_limit=(-25,15),val_shift_limit=(-15,15),p=p_soft),
    #     # A.CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=p_soft),
    #     # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=p_soft),
    #     # A.RandomBrightness(limit=0.1, p=p_soft),
    #     # A.RandomContrast(limit=0.1, p=p_soft),
    #     ])
        
    #DATA NORMALIZATION
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["dataset"]["mean"], std=cfg["dataset"]["stddev"]),
        transforms.Resize(size=(model.resize_param, model.resize_param))
    ])

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          pipeline_transform,
                                          transforms.Resize(size=(model.resize_param, model.resize_param))])
    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(model.resize_param, model.resize_param))])

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f'{total_params:,} total parameters.')

    total_trainable_params = sum(
        p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    # Loss function
    criterion = getattr(torch.nn, cfg['training']['criterion'])()

    # Optimizer
    optimizer = getattr(torch.optim, cfg['training']['optimizer'])
    optimizer = optimizer(model.net.parameters(), **cfg['training']['optimizer_args'])

    scheduler = getattr(torch.optim.lr_scheduler, cfg['training']['lr_scheduler'])
    scheduler = scheduler(optimizer, **cfg['training']['lr_scheduler_args'])

    # Initialize momentum_encoder with parameters of encoder.
    momentum_step(m=0)

    # Training
    # # use the configuration for the dataset
    # dataset_arguments = cfg['dataset']
    # dataset_train = SkinLesionDataset(dataset_set='train',
    #                                   dataset_mean=dataset_arguments['mean'],
    #                                   dataset_std=dataset_arguments['stddev'],
    #                                   transform=transform_train,
    #                                   seg_image=dataset_arguments['use_masks'])
    # dataset_val = SkinLesionDataset(dataset_set='val',
    #                                 dataset_mean=dataset_arguments['mean'],
    #                                 dataset_std=dataset_arguments['stddev'],
    #                                 transform=transform_val,
    #                                 seg_image=dataset_arguments['use_masks'])
    
    # datasets = {'train': dataset_train, 'val': dataset_val}

    # # use the configuration for the dataloader
    # dataloader_arguments = cfg['dataloaders']
    # dataloader_train = DataLoader(dataset_train,
    #                               batch_size=dataloader_arguments['batch_size'],
    #                               shuffle=True,
    #                               num_workers=dataloader_arguments['num_workers'],
    #                               pin_memory=True)
    # dataloader_valid = DataLoader(dataset_val,
    #                               batch_size=dataloader_arguments['batch_size'],
    #                               num_workers=dataloader_arguments['num_workers'],
    #                               pin_memory=True)
    # dataloaders = {'train': dataloader_train, 'val': dataloader_valid}

    # # loss function
    # if 'criterion_args' in cfg['training']:
    #     if cfg['training']['criterion_args'].get('weight') is not None:
    #         holder = cfg['training']['criterion_args']['weight'].copy()
    #         cfg['training']['criterion_args']['weight'] = torch.tensor(cfg['training']['criterion_args']['weight'],
    #                                                                    dtype=torch.float,
    #                                                                    device=device)
    #     criterion = getattr(torch.nn, cfg['training']['criterion'])(**cfg['training']['criterion_args'])
    # else:
    #     criterion = getattr(torch.nn, cfg['training']['criterion'])()

    # # Optimizer
    # optimizer = getattr(torch.optim, cfg['training']['optimizer'])
    # optimizer = optimizer(net.parameters(), **cfg['training']['optimizer_args'])

    # scheduler = getattr(torch.optim.lr_scheduler, cfg['training']['lr_scheduler'])
    # scheduler = scheduler(optimizer, **cfg['training']['lr_scheduler_args'])
    # # **d means "treat the key-value pairs in the dictionary as additional named arguments to this function call."
    # with open(config_path, "r") as ymlfile:
    #     cfg = yaml.safe_load(ymlfile)
    # train(net, datasets, dataloaders, criterion, optimizer, scheduler, cfg)


if __name__ == '__main__':
    main()
