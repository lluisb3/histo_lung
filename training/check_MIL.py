from pathlib import Path
import pandas as pd
import torch
from training.utils_trainig import yaml_load


thispath = Path(__file__).resolve()

datadir = Path(thispath.parent.parent / "data")


# Load PyTorch model
experiment_name = "MIL_optimNic"

modeldir = Path(thispath.parent.parent / "trained_models" / "MIL" / experiment_name)

# cfg = yaml_load(modeldir / f"config_{experiment_name}.yml")

# bestdir = Path(modeldir / cfg.dataset.magnification / cfg.model.model_name)
bestdir = Path(modeldir / "10" / "resnet34")

checkpoint = torch.load(bestdir / f"{experiment_name}.pt")

train_loss = checkpoint["train_loss"]
valid_loss = checkpoint["valid_loss"]
epoch = checkpoint["epoch"] + 1

print(f"Loaded encoder using as backbone resnet34 with a best loss in train"
      f"of {train_loss} and in validation of {valid_loss} at Epoch {epoch}")

predictions = pd.read_csv(bestdir / "training_predictions_best.csv", index_col=0)

print(predictions)