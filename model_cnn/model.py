import os

import torch
import torch.nn as nn
import torchvision
from constants import DEVICE, RUNTIME_STR
from torch.optim import SGD, lr_scheduler


def densenet121_model(user_args):
    model = torchvision.models.densenet121(weights=True)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_features, 2), nn.Sigmoid())
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = SGD(model.parameters(), lr=user_args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, criterion, optimizer, scheduler


def save_model(config, model, runtime_str=RUNTIME_STR):
    project_name = config.project_name
    save_dir = config.training.save_dir

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{project_name}_{runtime_str}.pth")
    torch.save(obj=model, f=save_path)

    return None
