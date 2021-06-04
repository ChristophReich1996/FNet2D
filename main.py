import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch_optimizer
import torchvision
import torchvision.transforms as transforms
import os

from fnet_2d import FNet2D
from metrics import Accuracy
from model_wrapper import ModelWrapper
from logger import Logger

# Set cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Set device
device: str = "cuda"

if __name__ == '__main__':
    # Init transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    # Init datasets
    training_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, download=True,
                                                    transform=transform_train)
    training_dataset = DataLoader(training_dataset, batch_size=512, shuffle=True, num_workers=20, pin_memory=True)
    test_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, download=True,
                                                transform=transform_test)
    test_dataset = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=20, pin_memory=True)
    # Init model
    model = FNet2D()
    print("# parameters", sum([p.numel() for p in model.parameters()]))
    # Model to device
    model.to(device)
    # Init data parallel
    model = nn.DataParallel(model)
    # Init optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-02, momentum=0.9, weight_decay=5e-4)
    # Init learning rate schedule
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200)
    # Init loss function
    loss_function = nn.CrossEntropyLoss()
    # Init model wrapper
    model_wrapper = ModelWrapper(model=model,
                                 optimizer=optimizer,
                                 loss_function=loss_function,
                                 training_dataset=training_dataset,
                                 test_dataset=test_dataset,
                                 validation_metric=Accuracy(),
                                 logger=Logger(experiment_path_extension=str(model.__class__.__name__)),
                                 device=device)
    # Perform training
    model_wrapper.train(epochs=250)
