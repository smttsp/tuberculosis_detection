import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from .constants import CLASS_NAMES, DatasetType


torch.manual_seed(0)


def get_train_transform():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(224, 224)),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_val_transform():
    return transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_test_transform():
    return transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_transform_main(dataset_type: DatasetType):
    if dataset_type == DatasetType.train:
        transform = get_train_transform()
    elif dataset_type == DatasetType.val:
        transform = get_val_transform()
    else:
        transform = get_test_transform()

    return transform


class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        image_dict,
        dataset_type: DatasetType,
        class_names=CLASS_NAMES,
    ):
        self.root_dir = root_dir
        self.images = image_dict
        self.class_names = class_names
        self.transform = get_transform_main(dataset_type)

    def __len__(self):
        return sum([len(self.images[c]) for c in self.class_names])

    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.root_dir, class_name, image_name)
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.class_names.index(class_name)


def split_image_list(
    root_dir,
    split_ratio=(0.7, 0.15, 0.15),
    class_names=CLASS_NAMES,
):
    if len(split_ratio) != 3:
        raise Exception("split ratios should be a list with 3 elements")

    image_dirs = {cls: os.path.join(root_dir, cls) for cls in class_names}

    train_images = {}
    val_images = {}
    test_images = {}

    for cls in class_names:
        images = [
            x for x in os.listdir(image_dirs[cls]) if x.lower().endswith("png")
        ]
        random.shuffle(images)  # Shuffle the images

        n_train = int(len(images) * split_ratio[0])
        n_val = int(len(images) * split_ratio[1])

        train_images[cls] = images[:n_train]
        val_images[cls] = images[n_train : n_train + n_val]
        test_images[cls] = images[n_train + n_val :]

    return train_images, val_images, test_images


def get_single_loader(root_dir, images, dataset_type, batch_size):
    a_set = ChestXRayDataset(root_dir, images, dataset_type=dataset_type)
    a_loader = DataLoader(
        a_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    return a_loader


def get_loaders_main(
    root_dir, split_ratio, class_names=CLASS_NAMES, batch_size=1
):
    train_images, val_images, test_images = split_image_list(
        root_dir, split_ratio, class_names
    )

    train_loader = get_single_loader(
        root_dir,
        train_images,
        dataset_type=DatasetType.train,
        batch_size=batch_size,
    )
    val_loader = get_single_loader(
        root_dir,
        val_images,
        dataset_type=DatasetType.val,
        batch_size=batch_size,
    )
    test_loader = get_single_loader(
        root_dir,
        test_images,
        dataset_type=DatasetType.test,
        batch_size=batch_size,
    )

    return train_loader, test_loader, val_loader
