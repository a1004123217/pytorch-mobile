from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
from models.imagenet.MobileNetV2_CBAM import MobileNetV2_CBAM
from models.imagenet.MobileNetV2_LC import MobileNetV2_LC
from models.imagenet.mobilenetv3 import MobileNetV3
from models.imagenet.ghostnet import GhostNet
from models.imagenet.AlexNetMini_LC import AlexNetMini_LC
from models.imagenet.Efficient import EfficientNet
from models.imagenet.ShuffleNetV2 import ShuffleNetV2_1
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile
import timm
from models.imagenet.regnet import RegNetY_200MF

parser = argparse.ArgumentParser(description='PyTorch convert to onnx')

parser.add_argument('--model_path', default='', type=str, metavar='MODEL_PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--dest_path', default='', type=str, metavar='DEST_PATH',
                    help='path to onnx save local (default: none)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture')
args = parser.parse_args()

if args.arch.startswith('MobileNetV2_CBAM'):
    model = MobileNetV2_CBAM()
    print("train MobileNetV2_CBAM\n")
elif args.arch.startswith('MobileNetV2_LC'):
    model = MobileNetV2_LC()
    print("train MobileNetV2_LC\n")
elif args.arch.startswith('MobileNetV3'):
    model = MobileNetV3(pretrained=True,model_path=args.model_path)
    print("train MobileNetV3\n")
elif args.arch.startswith('AlexNetMini_LC'):
    model = AlexNetMini_LC()
    print("train AlexNetMini_LC\n")
elif args.arch.startswith('ShuffleNetV2'):
    model = ShuffleNetV2_1(pretrained=True,model_path=args.model_path)
elif args.arch.startswith('EfficientNetB0'):
    model = EfficientNet.from_pretrained('efficientnet-b0',weights_path=args.model_path)
    print('train EfficientNetB0')
elif args.arch.startswith('EfficientNetB1'):
    model = EfficientNet.from_pretrained('efficientnet-b1',weights_path=args.model_path)
    print('train EfficientNetB1')
elif args.arch.startswith('EfficientNetB2'):
    model = EfficientNet.from_pretrained('efficientnet-b2',weights_path=args.model_path)
    print('train EfficientNetB2')
elif args.arch.startswith('EfficientNet-lite0'):
    model = timm.create_model('efficientnet_lite0',pretrained=True)
    print('train EfficientNet-lite0')
elif args.arch.startswith('RegNetY_200MF'):
    model = RegNetY_200MF('RegNetY_200MF',model_path=args.model_path)
    print('train RegNetY_200MF\n')
elif args.arch.startswith('RegNetY_400MF'):
    model = RegNetY_200MF('RegNetY_400MF',model_path=args.model_path)
    print('train RegNetY_400MF\n')
elif args.arch.startswith('RegNetY_600MF'):
    model = RegNetY_200MF('RegNetY_600MF',model_path=args.model_path)
    print('train RegNetY_600MF\n')
elif args.arch.startswith('RegNetY_800MF'):
    model = RegNetY_200MF('RegNetY_800MF',model_path=args.model_path)
    print('train RegNetY_800MF\n')
elif args.arch.startswith('GhostNet'):
    model = GhostNet(pretrained=True, model_path=args.model_path)
    print("train GhostNet\n")
else:
    print("=> creating model '{}'".format(args.arch))

#define input shape
x = torch.rand(1, 3, 224, 224)
#define input and output nodes, can be customized
input_names = ["input"]
output_names = ["cls"]
#model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))
model.eval()
#model.to("cuda")
#convert pytorch to onnx
torch_out = torch.onnx.export(model, x, args.dest_path, verbose=False, input_names=input_names, output_names=output_names)

print("\n model conver done.\n")

