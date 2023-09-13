import copy
import time

import torch
from constants import DEVICE, DatasetType
from model import save_model


torch.manual_seed(0)


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    config,
):
    user_args = config.training
    num_epochs = user_args.num_epochs

    since = time.time()
    tr = {DatasetType.train: train_loader, DatasetType.val: val_loader}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}", "\n", "-" * 10)

        # Each epoch has a training and validation phase
        for phase in [DatasetType.train, DatasetType.val]:
            is_training = phase == DatasetType.train
            if is_training:
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tr[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(is_training):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if is_training:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if is_training:
                scheduler.step()

            epoch_loss = running_loss / len(tr[phase].dataset)
            epoch_acc = running_corrects.double() / len(tr[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if not is_training and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_model(config, best_model_wts)

        print()

    print_elapsed_time(since)
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def print_elapsed_time(since):
    time_elapsed = time.time() - since
    minutes = time_elapsed // 60
    seconds = time_elapsed % 60
    print(f"Training complete in {minutes:.0f}m {seconds:.0f}s")
