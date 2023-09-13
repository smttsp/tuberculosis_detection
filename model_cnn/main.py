from datetime import datetime

import hydra
import torch
from constants import ROOT_DIR, DEVICE
from dataset import get_loaders_main
from model import densenet121_model
from training import train_model
from evaluation import pytorch_predict
import torchvision


torch.manual_seed(0)


@hydra.main(
    version_base=None, config_path="./configs", config_name="densenet121"
)
def main(config):
    print(f"TRAINING START TIME: {datetime.now().isoformat()}")

    user_args = config.training
    model, criterion, optimizer, scheduler = densenet121_model(user_args)

    train_loader, test_loader, val_loader = get_loaders_main(
        root_dir=ROOT_DIR,
        split_ratio=user_args.split_ratio,
        batch_size=user_args.batch_size,
    )
    model = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        config=config,
    )

    df1 = pytorch_predict(model, test_loader, device=DEVICE)
    print("test metrics:\n", df1.head())

    print(f"TRAINING END TIME: {datetime.now().isoformat()}")
    return model


if __name__ == "__main__":
    import torchvision
    import torch
    from evaluation import do_inference
    import torch.nn as nn

    model = torchvision.models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_features, 2), nn.Sigmoid())

    save_path = "saved_models/densenet121/best_model.pth"
    weights = torch.load(f=save_path)
    model.load_state_dict(weights)

    image_path = "/users/samet/Documents/data/TB_Chest_Radiography_Database/Normal/Normal-1821.png"
    x = do_inference(model, image_path=image_path)

    densenet_121 = main()
    print("finito")
