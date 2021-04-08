import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import pickle
import random
from PIL import ImageOps, ImageFilter

class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_path, img_filename, label_filename, transform=None, train=False):
        self.img_path = os.path.join(data_path, img_path)
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        labels = np.loadtxt(label_filepath, dtype=np.int64)
        self.label = labels
        self.train = train

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        #if self.transform is not None:
        if self.train:
            img1, img2 = self.transform(img)
        else:
            img = self.transform(img)

        label = torch.from_numpy(self.label[index])
        label = label.type(torch.FloatTensor)
        if self.train:
            return (img1, img2), label
        else:
            return img, label

    def __len__(self):
        return len(self.img_filename)

def split_idxs(pkl_file, percent):

    train_ids = pickle.load(open(pkl_file, 'rb'))
    partial_size = int(percent*len(train_ids))

    labeled_idxs = train_ids[:partial_size]
    unlabeled_idxs = train_ids[partial_size:]

    return labeled_idxs, unlabeled_idxs

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TransformTwice:
    def __init__(self, transform, aug_transform):
        self.transform = transform
        self.aug_transform = aug_transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.aug_transform(inp)
        return out1, out2

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
