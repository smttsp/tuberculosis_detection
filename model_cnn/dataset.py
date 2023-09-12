import os
import random

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


torch.manual_seed(0)


class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        self.images = {}
        self.class_names = ["Normal", "Tuberculosis"]
        self.image_dirs = {
            cls: os.path.join(root_dir, cls) for cls in self.class_names
        }
        # image_dirs = {
        #     'Normal': root_dir + '/Normal',
        #     'Tuberculosis': root_dir + '/Tuberculosis'
        # }
        #
        # self.image_dirs = image_dirs

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


def split_dataset(dataset, split_ratios):
    if len(split_ratios) != 3:
        raise Exception("split ratios should be a list with 3 elements")
    if not len(dataset):
        raise Exception("Dataset cannot be empty")

    data_sizes = [
        int(s / sum(split_ratios) * len(dataset)) for s in split_ratios
    ]

    train_set, test_set, val_set = random_split(dataset, data_sizes)
    return train_set, test_set, val_set


def get_loaders(dataset, split_ratios, batch_size=1):
    train_set, test_set, val_set = split_dataset(dataset, split_ratios)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader, val_loader
