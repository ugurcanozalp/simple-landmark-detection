
import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from module import LandmarkDetector
from datasets import AnimalPoseLandmarkDataset

parser = ArgumentParser()
parser.add_argument('--test', action="store_true")
parser.add_argument('--ckpt', type=str, default="checkpoints/convnext_tiny.pt.ckpt")
parser.add_argument('--batch_size', type=int, default=16)

parser = LandmarkDetector.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
dict_args = vars(args)
model = LandmarkDetector(**dict_args)
# define checkpoint callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="./checkpoints",
    verbose=True,
    filename=args.configuration+".pt",
    monitor="l1_errors",
    save_top_k=1,
    save_weights_only=True,
    mode="min" # only pick min of `l1_errors`
)

trainer = pl.Trainer(
    gpus=args.gpus,
    callbacks=[checkpoint_callback],
    gradient_clip_val=args.gradient_clip_val)

train_ds = AnimalPoseLandmarkDataset(path="data/animalpose", split="augment") 

train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
    num_workers=10)

val_ds = AnimalPoseLandmarkDataset(path="data/animalpose", split="test") 

val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
    num_workers=1)

test_ds = AnimalPoseLandmarkDataset(path="data/animalpose", split="test") 

test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
    num_workers=1)

if args.test:
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state_dict"])
    trainer.test(model, test_dataloaders=[test_dl])
else:
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state_dict"])
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=[val_dl])

# python animalpose.py --arch convnext --configuration convnext_tiny --num_landmarks 20 --learning_rate 2e-4 --weight_decay 0 --batch_size 32 --max_epochs 10 --min_epochs 3 --gpus 1

# --gradient_clip_val 1.0