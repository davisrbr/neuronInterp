import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Compose, Normalize
from torchvision.transforms.functional import to_tensor
import numpy as np
from tqdm.notebook import tqdm
from IPython.display import Image, HTML, clear_output
import matplotlib.pyplot as plt
import seaborn as sns


# train loop
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    losses = []
    epoch_fracs = []
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            losses.append(loss.item())
            epoch_frac = (epoch - 1) + batch_idx / len(train_loader)
            epoch_fracs.append(epoch_frac)
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
    return epoch_fracs, losses


# test loop
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.sum(criterion(output, target)).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return  test_loss, 100. * correct / len(test_loader.dataset)


def run_train_test_loop(model: torch.nn.Module, train_loader, test_loader, model_name: str, epochs: int = 20, device: torch.device = torch.device("cpu")):
    sns.set_style("darkgrid")
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    val_best = 0.
    val_accs = []
    val_losses = []
    train_losses = []
    train_iters = []
    val_epochs = []
    for epoch in tqdm(range(1, epochs + 1)):
        epoch_fracs, train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = test(model, device, test_loader, criterion)
        scheduler.step()

        train_iters.extend(epoch_fracs)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        train_losses.extend(train_loss)
        val_epochs.append(epoch)

        clear_output(True)
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        ax1.set_yscale('log')
        lns1_1 = ax1.plot(train_iters, train_losses, ".", label="train loss", color="y")
        lns1_0 = ax1.plot(val_epochs, val_losses, "o", label="validation loss")
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        lns2 = ax2.plot(val_epochs, val_accs, "o", label="validation accuracy", color=color)
        ax2.set_ylabel('accuracy',)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        leg = lns1_0 + lns1_1 + lns2
        labs = [l.get_label() for l in leg]
        ax1.legend(leg, labs, loc=0)
        plt.show()

        if val_acc > val_best:
            val_best = val_acc
            torch.save(model.state_dict(), f"models/{model_name}.pt")
    print(f"best acc: {val_best}")
