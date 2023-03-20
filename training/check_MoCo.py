from pathlib import Path
import torch
from training import Encoder, ModelOption, yaml_load

thispath = Path(__file__).resolve()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

experiment_name = "MoCo_try_Adam"

modeldir = Path(thispath.parent.parent / "trained_models" / experiment_name)

cfg = yaml_load(modeldir / f"config_{experiment_name}.yml")

model = ModelOption(cfg.model.model_name,
                cfg.model.num_classes,
                freeze=cfg.model.freeze_weights,
                num_freezed_layers=cfg.model.num_frozen_layers,
                dropout=cfg.model.dropout,
                embedding_bool=cfg.model.embedding_bool
                )    

    
# Encoder and momentum encoder
moco_dim = cfg.training.moco_dim

encoder = Encoder(model, dim=moco_dim).to(device)

optimizer = getattr(torch.optim, cfg.training.optimizer)
optimizer = optimizer(encoder.parameters(), **cfg.training.optimizer_args)

# checkpoint = torch.load(modeldir / cfg.dataset.magnification / cfg.model.model_name / f"{experiment_name}.pt")
checkpoint = torch.load(modeldir / cfg.dataset.magnification / cfg.model.model_name / "MoCo.pt")
encoder.load_state_dict(checkpoint["encoder_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
loss = checkpoint["loss"]
epoch = checkpoint["epoch"] + 1

print(f"Loaded encoder using as backbone {cfg.model.model_name} with a best loss of {loss} at Epoch {epoch}")
print(optimizer.param_groups[0]['lr'])

