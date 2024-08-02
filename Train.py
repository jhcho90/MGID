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
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader_DDN
from model.Reconstruction_skip import *
from utils import *
import random
from skimage import io
import glob
from SSIM import SSIM


import argparse

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.01, help='weight of the feature compactness loss')
parser.add_argument('--loss_white', type=float, default=0.01, help='weight of the white loss')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--fdim', type=int, default=64, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=64, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=20, help='number of the memory items')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_path', type=str, default='/home/diml/Desktop/Code/Derain/siamese/data/trainingset2(synthetic)/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log_100', help='directory of log')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

torch.backends.cudnn.enabled = True

# Loading dataset
#train_dataset = DataLoader_DDN(args.dataset_path + '/Train/', transforms.Compose([transforms.ToTensor(),]))
train_dataset = DataLoader_DDN(args.dataset_path , transforms.Compose([transforms.ToTensor(),]))
train_size = len(train_dataset)
train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)

#test_dataset = DataLoader_DDN(args.dataset_path + '/Test/', transforms.Compose([transforms.ToTensor(),]))
test_dataset = DataLoader_DDN(args.dataset_path, transforms.Compose([transforms.ToTensor(),]))
test_size = len(train_dataset)
test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)


model = convAE(args.c, args.msize, args.fdim, args.mdim)
params_encoder = list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
model.cuda()

log_dir = './logdir_100/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    os.makedirs(os.path.join(log_dir, 'result'))


m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda()
prevModels = sorted(glob.glob(log_dir+'/*.pth'))
if len(prevModels) != 0:
    recent = prevModels[-1]
    prevEpoch = int(recent.split('_')[-1].split('.')[0])
    checkpoint = torch.load(os.path.join(recent), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    m_items = checkpoint['memory_items'].cuda()
    print('Checkpoint with epoch {} loaded!'.format(prevEpoch))
else:
    prevEpoch = -1
    print('Start training from beginning')

loss_func_mse = nn.MSELoss(reduction='none')
l1_function = nn.L1Loss().to()
criterion = SSIM().cuda()


for epoch in range(prevEpoch+1, args.epochs):
    model.train()
    start = time.time()
    # for j,(rain1, rain2) in enumerate(train_batch):
    #     rain1 = rain1.cuda()
    #     rain2 = rain2.cuda()

    for j, (rain) in enumerate(train_batch):
        rain = rain.cuda()
        rain1 = rain[:, :, :, :256]
        rain2 = rain[:, :, :, 256:]

        derained1, derained2, streak1, streak2, _, _, m_items, _, _, compactness_loss, loss_white = model.forward(rain1, rain2, m_items, True)


        optimizer.zero_grad()
        loss_pixel1 = l1_function(derained1, derained2)

        loss_tex1 = l1_function(derained1, rain2)
        loss_tex2 = l1_function(rain1, derained2)

        loss_same1 = l1_function(derained2 + streak1, rain1)
        loss_same2 = l1_function(derained1 + streak2, rain2)

        pixel_metric = criterion(derained1, derained2)
        ssim_loss = -pixel_metric


        loss = loss_pixel1  + loss_tex1 + loss_tex2 + loss_same1 + loss_same2 + args.loss_compact * compactness_loss + args.loss_white * loss_white + 0.5* ssim_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        print('{}, Loss: Reconstruction1 {:.6f} / Reconstruction2 {:.6f} / Compactness {:.6f} / White {:.6f} / SSIM {:.4f}'.format(j, loss_pixel1.item(), loss_tex1.item(), compactness_loss.item(), loss_white.item(), ssim_loss.item()))

        if j % 1000 == 0:
            in_rain1 = (np.transpose(rain1[0, :, :, :].data.cpu().numpy(), [1, 2, 0]) + 1.) / 2.
            derained_out1 = (np.transpose(derained1[0, :, :, :].data.cpu().numpy(), [1, 2, 0]) + 1.) / 2.
            #ground_truth1 = (np.transpose(GT1[0, :, :, :].data.cpu().numpy(), [1, 2, 0]) + 1.) / 2.
            in_rain2 = (np.transpose(rain2[0, :, :, :].data.cpu().numpy(), [1, 2, 0]) + 1.) / 2.
            derained_out2 = (np.transpose(derained2[0, :, :, :].data.cpu().numpy(), [1, 2, 0]) + 1.) / 2.
            #ground_truth2 = (np.transpose(GT2[0, :, :, :].data.cpu().numpy(), [1, 2, 0]) + 1.) / 2.
            concat_in_out1 = np.concatenate((in_rain1,derained_out1), axis=1)
            concat_in_out2 = np.concatenate((in_rain2, derained_out2), axis=1)
            concat_in_out = np.concatenate((concat_in_out1, concat_in_out2), axis=0)
            cv2.imwrite(os.path.join(log_dir,('result/train_%s_%s.png'%(epoch, j))), concat_in_out * 255)


    
    print('----------------------------------------')
    print('Epoch:', epoch)
    print('Loss: Reconstruction1 {:.6f} / Reconstruction2 {:.6f} / Compactness {:.6f}'.format(loss_pixel1.item(), loss_tex1.item(),compactness_loss.item()))
    # print('Memory_items:')
    # print(m_items)
    print('----------------------------------------')

    if (epoch % 2) ==0:
        save_on_master({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'memory_items': m_items,
            'args': args},
            os.path.join(log_dir, 'model_{0:05}.pth'.format(epoch)))
