import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from collections import defaultdict
import os
# from cv2 import imread, createCLAHE
import cv2
from glob import glob

import matplotlib.pyplot as plt
from keras.models import model_from_json
import shutil
import random
import torchvision
import time
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim import lr_scheduler

import argparse
torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as F
from os.path import isfile
from os import rename

from sklearn.metrics import auc, roc_curve
from PIL.ImageFilter import GaussianBlur

from numpy import pi as PI
from numpy import sqrt
from scipy.special import comb

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import helper
import copy
from sklearn.metrics import confusion_matrix
from PIL import Image
from matplotlib.pyplot import imshow


class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('png')]
            print(f'Found {len(images)}{class_name}')
            return images

        self.images = {}
        self.class_names = ['Normal', 'Tuberculosis']
        for c in self.class_names:
            self.images[c] = get_images(c)
        self.image_dirs = image_dirs
        self.transform = transform

    def __len__(self):
        return sum([len(self.images[c]) for c in self.class_names])

    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)

def get_transformation():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size=(224, 224)),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    valid_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(20),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, valid_transform