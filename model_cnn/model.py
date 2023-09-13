import torch.nn as nn
from constants import DEVICE
from torch.optim import SGD, lr_scheduler
import torchvision
# from torchvision import models


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
