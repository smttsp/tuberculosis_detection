import argparse
import os
import random
import re
import shutil
import time
from collections import defaultdict
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm


# from cv2 import imread, createCLAHE


torch.manual_seed(0)

# from os import rename
# from os.path import isfile
#
# import torch.nn as nn
# import torch.nn.functional as F
# from numpy import pi as PI, sqrt
# from PIL.ImageFilter import GaussianBlur
# from scipy.special import comb
# from sklearn.metrics import auc, roc_curve


# from keras.models import model_from_json
# import tensorflow as tf
# from tensorflow import keras
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import load_model
# import helper


class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        self.images = {}
        self.class_names = ["Normal", "Tuberculosis"]
        self.image_dirs = image_dirs

        for c in self.class_names:
            self.images[c] = self.get_images(c)
        self.transform = transform

    def __len__(self):
        return sum([len(self.images[c]) for c in self.class_names])

    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.class_names.index(class_name)

    def get_images(self, class_name):
        images = [
            x
            for x in os.listdir(self.image_dirs[class_name])
            if x.lower().endswith("png")
        ]
        print(f"Found {len(images)}{class_name}")
        return images


def get_transformation():
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(224, 224)),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    valid_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(20),
            torchvision.transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, valid_transform
