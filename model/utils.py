# import numpy as np
# from collections import OrderedDict
# import os
# import glob
# import cv2
# import torch.utils.data as data
# import random
#
# rng = np.random.RandomState(2017)
#
# def np_load_frames(rain_pair):
#     rain1 = cv2.imread(rain_pair[0])
#     rain2 = cv2.imread(rain_pair[1])
#     #gt = cv2.imread(name_gt)
#     # transformed = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2YCR_CB)
#     # transformed = transformed[:, :, 0]
#     height_orig, width_orig = rain1.shape[:2]
#     height_resize = int(np.round(height_orig / 32) * 32)
#     width_resize = int(np.round(width_orig / 32) * 32)
#
#     rain_resized1 = cv2.resize(rain1, (256, 256))
#     rain_resized1 = rain_resized1.astype(dtype=np.float32)
#     rain_resized1 = (rain_resized1 / 127.5) - 1.0
#     rain_resized2 = cv2.resize(rain2, (256, 256))
#     rain_resized2 = rain_resized2.astype(dtype=np.float32)
#     rain_resized2 = (rain_resized2 / 127.5) - 1.0
#
#     #gt_resized = cv2.resize(gt, (512, 512))
#     #gt_resized = gt_resized.astype(dtype=np.float32)
#     #gt_resized = (gt_resized / 127.5) - 1.0
#
#     return rain_resized1, rain_resized2 #, gt_resized
#
#
# class DataLoader_DDN(data.Dataset):
#     def __init__(self, dir, transform):
#         self.dir = dir
#         self.transform = transform
#         self.rain_images = glob.glob(self.dir + '/*')
#         self.rain_pairs = []
#         for i in range(900):
#             for j in range(14):
#                 for k in range(14-j-1):
#                     rain_pair = []
#                     rain_pair.append(dir + '/{}_{}.jpg'.format(i+1, j+1))
#                     rain_pair.append(dir + '/{}_{}.jpg'.format(i+1, 14-k))
#                     self.rain_pairs.append(rain_pair)
#                     rain_pair = []
#
#     def __getitem__(self, index):
#         rain1, rain2 = np_load_frames(self.rain_pairs[index])
#         #rain2, norain2 = np_load_frames(self.rain_images[self.__len__() - index - 1], self.clean_images[self.__len__() - index - 1])
#         return self.transform(rain1), self.transform(rain2) #, self.transform(norain1), self.transform(rain2), self.transform(norain2)
#
#     def __len__(self):
#         return len(self.rain_pairs)


import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import random

rng = np.random.RandomState(2017)

def np_load_frames(name_rain):
    rain = cv2.imread(name_rain)
    #gt = cv2.imread(name_gt)
    # transformed = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2YCR_CB)
    # transformed = transformed[:, :, 0]
    height_orig, width_orig = rain.shape[:2]
    height_resize = int(np.round(height_orig / 32) * 32)
    width_resize = int(np.round(width_orig / 32) * 32)

    rain_resized = cv2.resize(rain, (512, 256))
    rain_resized = rain_resized.astype(dtype=np.float32)
    rain_resized = (rain_resized / 127.5) - 1.0

    #gt_resized = cv2.resize(gt, (512, 512))
    #gt_resized = gt_resized.astype(dtype=np.float32)
    #gt_resized = (gt_resized / 127.5) - 1.0

    return rain_resized #, gt_resized


class DataLoader_DDN(data.Dataset):
    def __init__(self, dir, transform):
        self.dir = dir
        self.transform = transform
        self.rain_images = glob.glob(self.dir + '/*')
        self.clean_images = []
        for i in range(len(self.rain_images)):
            rain_im_name = self.rain_images[i].split('/')[-1]
            #clean_im_name = rain_im_name.split('_')[0] + '.jpg'
            self.clean_images.append(self.dir + '/' +rain_im_name)

    def __getitem__(self, index):
        rain1 = np_load_frames(self.rain_images[index])
        #rain2, norain2 = np_load_frames(self.rain_images[self.__len__() - index - 1], self.clean_images[self.__len__() - index - 1])
        return self.transform(rain1) #, self.transform(norain1), self.transform(rain2), self.transform(norain2)

    def __len__(self):
        return len(self.rain_images)