import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Memory import *
from .ISW_loss import *

class Encoder(torch.nn.Module):
    def __init__(self, n_channel=3):
        super(Encoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )

        self.moduleConv1 = Basic(n_channel, 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 64)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 64)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(64, 64)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(64)
        self.moduleReLU = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)

        return tensorConv1, tensorConv2, tensorConv3, tensorConv4


class Decoder(torch.nn.Module):
    def __init__(self, n_channel=3):
        super(Decoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh())

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv = Basic(64 + 64, 64)
        self.moduleUpsample4 = Upsample(64, 64)

        self.moduleDeconv3 = Basic(64 + 64, 64)
        self.moduleUpsample3 = Upsample(64, 64)

        self.moduleDeconv2 = Basic(64 + 64, 64)
        self.moduleUpsample2 = Upsample(64, 64)

        self.moduleDeconv1 = Gen(64 + 64, n_channel, 64)

    def forward(self, x1, x2, x3, x4):
        tensorConv = self.moduleConv(x4)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)

        tensorConcat3 = torch.cat([x3, tensorUpsample4], dim=1)
        tensorDeconv3 = self.moduleDeconv3(tensorConcat3)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorConcat2 = torch.cat([x2, tensorUpsample3], dim=1)
        tensorDeconv2 = self.moduleDeconv2(tensorConcat2)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorConcat1 = torch.cat([x1, tensorUpsample2], dim=1)
        output = self.moduleDeconv1(tensorConcat1)

        #return output, tensorUpsample4, tensorUpsample3, tensorUpsample2
        return output


class convAE(torch.nn.Module):
    def __init__(self, n_channel=3, memory_size=10, feature_dim=512, key_dim=512, temp_update=0.1, temp_gather=0.1):
        super(convAE, self).__init__()

        self.encoder = Encoder(n_channel)
        self.decoder = Decoder(n_channel)
        self.memory = Memory(memory_size, feature_dim, key_dim, temp_update, temp_gather)
        self.instNorm = torch.nn.InstanceNorm2d(feature_dim)

    def forward(self, x1, x2, keys,train=True):

        if train:
            fea1_1, fea2_1, fea3_1, fea4_1 = self.encoder(x1)
            fea1_2, fea2_2, fea3_2, fea4_2 = self.encoder(x2)
            fea4_1 = self.instNorm(fea4_1)
            fea4_2 = self.instNorm(fea4_2)


            #whitening_loss
            loss_white = ISWloss(fea4_1, fea4_2)

            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(fea4_1, keys, train)
            updated_fea2, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(fea4_2, keys, train)


            streak1 = self.decoder(fea1_1, fea2_1, fea3_1, updated_fea)
            x1 = x1 - torch.tanh(streak1)
            streak2 = self.decoder(fea1_2, fea2_2, fea3_2, updated_fea2)
            x2 = x2 - torch.tanh(streak2)

            return x1, x2, torch.tanh(streak1), torch.tanh(streak2), fea4_1, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, loss_white



        else:

            fea1, fea2, fea3, fea4 = self.encoder(x1)
            fea42 = fea4
            fea4 = self.instNorm(fea4)
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss = self.memory(fea4, keys, train)
            #streak1, tensorUpsample4, tensorUpsample3, tensorUpsample2 = self.decoder(fea1, fea2, fea3, updated_fea)
            streak1 = self.decoder(fea1, fea2, fea3, updated_fea)
            x1 = x1 - torch.tanh(streak1)

            #return x1, torch.tanh(streak1), fea4, fea3, fea2, fea1, tensorUpsample4, tensorUpsample3, tensorUpsample2, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss
            return x1, torch.tanh(streak1), fea42, fea2, fea1, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss

