from argparse import ArgumentParser
import json
import os
from tqdm import tqdm
from shutil import copyfile
import numpy as np
import pytorch_lightning as pl
import torch

from module import LandmarkDetector

parser = ArgumentParser()
parser.add_argument('--arch', type=str, default='convnext')
parser.add_argument('--configuration', type=str, default='convnext_tiny')
parser.add_argument('--num_landmarks', type=int, default=20)
parser.add_argument('--ckpt', type=str, default="checkpoints/convnext_tiny.pt-v1.ckpt")
parser.add_argument('--quantize', action='store_true')

args = parser.parse_args()
model = LandmarkDetector(arch = args.arch, configuration= args.configuration, num_landmarks=args.num_landmarks)
sd = torch.load(args.ckpt, map_location="cpu")
model.load_state_dict(sd["state_dict"])
model.eval()

inp = (torch.rand(1, 3, 96, 128),)

deployment_path = os.path.join("deployment", args.configuration)
os.mkdir(deployment_path)
onnx_path = os.path.join(deployment_path, args.configuration+".onnx")
model.to_onnx(
    onnx_path, inp, export_params=True,
    opset_version=11,
    input_names = ['image'],   # the model's input names
    output_names = ['landmark'], # the model's output names
    dynamic_axes={'image' : {0 : 'batch_size'}}
)

if args.quantize:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quant_path = os.path.join(deployment_path, args.configuration+"-quantized.onnx")
    quantized_model = quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QUInt8)
