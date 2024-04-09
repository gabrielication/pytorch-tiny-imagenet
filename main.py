
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models.vgg import VGG16_BN_Weights, VGG19_BN_Weights
import torchvision.transforms as transforms

from test_model import test_model
from train_model import train_model
import argparse

from multiprocessing import freeze_support

torch.manual_seed(42)

def main():
    if torch.cuda.is_available():
        device = "cuda:0"
        num_gpus = torch.cuda.device_count()
    elif torch.backends.mps.is_built():
        device = torch.device("mps")
        num_gpus = 1
    else:
        device = "cpu"
        num_gpus = 0

    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs.")
    elif num_gpus == 1:
        print("Using 1 GPU.")
    else:
        print("CUDA is not available. Using CPU.")
        
    print(f"device: {device}")
        
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PyTorch Tiny ImageNet Training")
    parser.add_argument("--arch", type=str, default=None, choices=["vgg16", "vgg19", "resnet"], help="model architecture (default: vgg16)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs for training")
    args = parser.parse_args()

    data_dir = "tiny-224/"
    num_workers = {"train": 4, "val": 0, "test": 0}
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
    }

    print(f"batch size: {args.batch_size}")

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val", "test"]
    }
    dataloaders = {
        x: data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=num_workers[x])
        for x in ["train", "val", "test"]
    }

    if args.arch:
        archs = [args.arch]
    else:
        archs = ["vgg16", "vgg19", "resnet"]
        
    best_weights = []

    for arch in archs:
        
        print(f"architecture: {arch}")
        
        if(arch == "resnet"):
            print("currently not supported")
            continue
        
            # model_ft = models.resnet50(weights="IMAGENET1K_V1")
            # # Finetune Final few layers to adjust for tiny imagenet input
            # model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
            # num_features = model_ft.fc.in_features
            # model_ft.fc = nn.Linear(num_features, 200)
        elif(arch == "vgg16"):
            # Load VGG16
            model = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
            
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            # The output of AdaptiveAvgPool2d will be a fixed size (512 x 1 x 1)
            # This allows us to redefine the classifier part without worrying about the input size
            model.classifier = nn.Sequential(
                # Redefine the fully connected layers
                nn.Linear(512, 4096),  # Input features now fixed to 512
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 200)  # Assuming 200 classes for Tiny ImageNet
            )
            
            model_ft = model
            
        elif(arch == "vgg19"):
            # Load VGG19
            model = models.vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
            
            print(model)
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            # The output of AdaptiveAvgPool2d will be a fixed size (512 x 1 x 1)
            # This allows us to redefine the classifier part without worrying about the input size
            model.classifier = nn.Sequential(
                # Flatten the output of the adaptive avg pool
                # Redefine the fully connected layers
                nn.Linear(512, 4096),  # Input features now fixed to 512
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 200)  # Assuming 200 classes for Tiny ImageNet
            )

        if num_gpus > 1:
            model_ft = nn.DataParallel(model_ft)
        model_ft = model_ft.to(device)

        # Loss Function
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Train
        best_epoch = train_model(
            output_path=f"{arch}_224",
            model=model_ft,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer_ft,
            device=device,
            num_epochs=args.epochs,
        )

        best_weight = f"models/{arch}_224/model_{best_epoch}_epoch.pt"
        best_weights.append(best_weight)
        
        # Test
        model_ft.load_state_dict(torch.load(best_weight))
        test_model(model=model_ft, dataloaders=dataloaders, criterion=criterion, device=device)

    # Create a text file to store the best weights
    with open("best_weights.txt", "w") as file:
        # Write each best weight on a new line
        for weight in best_weights:
            file.write(weight + "\n")

if __name__ == '__main__':
    freeze_support()
    main()