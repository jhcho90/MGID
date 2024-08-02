import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils_test import DataLoader_DDN
from model.Reconstruction_skip import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
from math import log10
from skimage import io

import argparse


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.01, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.01, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=2, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=64, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=64, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=30, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.7, help='weight for the anomality score')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='JORDER_L', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--model_dir', type=str, default= './logdir/model_00010.pth', help='directory of model')
#parser.add_argument('--m_items_dir', type=str, default= './logdir/key_00010.pt', help='directory of model')
args = parser.parse_args()

def psnr(img1, img2):
    mse = np.mean((img1-img2) **2)
    return 10 * log10(1./mse)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]


torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

test_folder = "/media/jh/새 볼륨/RAIN_all_thing/Rain/Time-lapse_training_data/trainingset2(synthetic)" #args.dataset_path+args.dataset_type +"/training/frames"

# Loading dataset
test_dataset = DataLoader_DDN(test_folder,  transforms.Compose([transforms.ToTensor(),]))
test_size = len(test_dataset)
test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)


log_dir = os.path.join('./results' , args.dataset_type )
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
params_encoder =  list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
model.cuda()

model.load_state_dict(torch.load(args.model_dir)['model'])
model.to() # cuda()
m_items = torch.load(args.model_dir)['memory_items']
#m_items = m_items['model']
m_items_test = m_items.clone()
model.eval()




for k,(rain) in enumerate(test_batch):

    path = rain[1]

    rain = rain[0].cuda()
    rain1 = rain[:, :, :, :256]
    rain2 = rain[:, :, :, 256:]


    derained1, derained2, streak1, streak2, _, _, m_items, _, _, compactness_loss = model.forward(rain1, rain2, m_items,
                                                                                                  True)

    in_rain1 = (np.transpose(rain1[0, :, :, :].data.cpu().numpy(), [1, 2, 0]) + 1.) / 2.
    derained_out1 = (np.transpose(derained1[0, :, :, :].data.cpu().numpy(), [1, 2, 0]) + 1.) / 2.
    streak1 = (np.transpose(streak1[0, :, :, :].data.cpu().numpy(), [1, 2, 0]) + 1.) / 2.

    in_rain2 = (np.transpose(rain2[0, :, :, :].data.cpu().numpy(), [1, 2, 0]) + 1.) / 2.
    derained_out2 = (np.transpose(derained2[0, :, :, :].data.cpu().numpy(), [1, 2, 0]) + 1.) / 2.
    streak2 = (np.transpose(streak2[0, :, :, :].data.cpu().numpy(), [1, 2, 0]) + 1.) / 2.
    concat_in_out1 = np.concatenate((in_rain1, derained_out1, streak1), axis=1)
    concat_in_out2 = np.concatenate((in_rain2, derained_out2, streak2), axis=1)
    concat_in_out = np.concatenate((concat_in_out1, concat_in_out2), axis=0)

    cv2.imwrite(os.path.join(log_dir, path[0].split('/')[-1]), concat_in_out * 255)
