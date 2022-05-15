
import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from module import LandmarkDetector
from datasets import CustomCowLandmarkDataset

parser = ArgumentParser()
parser.add_argument('--test', action="store_true")
parser.add_argument('--ckpt', type=str, default="checkpoints/resnet18.pt.ckpt")
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

train_ds = CustomCowLandmarkDataset(path="data/customcow", split="train") 

train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
    num_workers=0)

val_ds = CustomCowLandmarkDataset(path="data/customcow", split="val") 

val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
    num_workers=0)

test_ds = CustomCowLandmarkDataset(path="data/customcow", split="test") 

test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
    num_workers=0)

if args.test:
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state_dict"])
    trainer.test(model, test_dataloaders=[test_dl])
else:
    trainer.fit(model, train_dataloader=train_dl, val_dataloaders=[val_dl])

# python cow.py --arch resnet --configuration resnet18 --learning_rate 1e-3 --weight_decay 1e-4 --batch_size 32 --max_epochs 10 --min_epochs 3 --gpus 1

# --gradient_clip_val 1.0