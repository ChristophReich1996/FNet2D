import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch_optimizer
import torchvision
import torchvision.transforms as transforms
import os

from fnet_2d import FNet2D
from utils import progress_bar

# Set cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Set device
device: str = "cuda"

# Init best accuracy
best_accuracy: float = 0.

# Init transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4, padding_mode="edge"),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.1),
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
model_name = "FNet2D"
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


def train(epoch: int) -> None:
    """
    Training function
    :param epoch: (int) Number of the current epoch
    """
    # Print epoch
    print("Epoch: {}".format(epoch))
    # Model into train mode
    model.train()
    # Init log variables
    training_loss: float = 0.
    correct_predictions: int = 0
    total: int = 0
    # Training loop
    for index, (inputs, labels) in enumerate(training_dataset):
        # Data to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Make prediction
        prediction = model(inputs)
        # Calc loss
        loss = loss_function(prediction, labels)
        # Compute gradients
        loss.backward()
        # Perform optimization
        optimizer.step()
        # Update logs
        training_loss += loss.item()
        total += prediction.shape[0]
        class_prediction = prediction.softmax(dim=-1).argmax(dim=-1)
        correct_predictions += (class_prediction == labels).sum().item()
        # Print progress bar
        progress_bar(current=index, total=len(training_dataset),
                     msg="Loss: {:.4f} | Acc: {:.4f} ({}/{})".format(training_loss / float(index + 1.),
                                                                     100. * float(correct_predictions) / float(total),
                                                                     correct_predictions, total))


def test(epoch: int) -> None:
    """
    Test function
    :param epoch: (int) Number of the current epoch
    """
    # Print info
    print("Testing")
    # Init log variables
    global best_accuracy
    test_loss: float = 0.
    correct_predictions: int = 0
    total: int = 0
    # Model into eval mode
    model.eval()
    # Compute no gradients
    with torch.no_grad():
        # Training loop
        for index, (inputs, labels) in enumerate(test_dataset):
            # Data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Make prediction
            prediction = model(inputs)
            # Compute loss
            loss = loss_function(prediction, labels)
            # Update logs
            test_loss += loss.item()
            total += prediction.shape[0]
            class_prediction = prediction.softmax(dim=-1).argmax(dim=-1)
            correct_predictions += (class_prediction == labels).sum().item()
            # Print progress bar
            progress_bar(current=index, total=len(test_dataset),
                         msg="Loss: {:.4f} | Acc: {:.4f} ({}/{})".format(
                             test_loss / float(index + 1.),
                             100. * float(correct_predictions) / float(total),
                             correct_predictions, total))
    # Save model
    accuracy = 100. * float(correct_predictions) / float(total)
    if accuracy > best_accuracy:
        # Set new best accuracy
        best_accuracy = accuracy
        # Print info
        print("Save best model with accuracy", accuracy)
        # Make folder
        if not os.path.isdir(model_name):
            os.makedirs(model_name)
        # Save model
        torch.save({"model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    "acc": accuracy,
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict()},
                   "./{}/checkpoint.pt".format(model_name))


if __name__ == '__main__':
    for epoch in range(250):
        train(epoch=epoch)
        test(epoch=epoch)
        lr_schedule.step()
