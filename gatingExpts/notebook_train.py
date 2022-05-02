import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Compose, Normalize
from torchvision.transforms.functional import to_tensor
import numpy as np
#from tqdm.notebook import tqdm
from IPython.display import Image, HTML, clear_output
import matplotlib.pyplot as plt
import seaborn as sns


# train loop
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    losses = 0.0
    epoch_fracs = []
    n = 0
    correct = 0.0
    #print('b2', flush=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        #print(batch_idx, 'b212')
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = criterion(output, target)
        #print(batch_idx, 'b213')
        loss.backward()
        optimizer.step()
        losses += loss.item()*target.shape[0]
        #print(batch_idx, 'b214')
    #print('b22')
    return losses / len(train_loader.dataset), 100. * correct / len(train_loader.dataset)




# test loop
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    #print('b3')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            #print(batch_idx, 'b31', flush=True)
            data, target = data.to(device), target.to(device)
            #print(batch_idx, 'b32', flush=True)
            output = model(data)
            #print(batch_idx, 'b33', flush=True)
            test_loss += criterion(output, target).item()*target.shape[0]  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            #print(batch_idx, 'b34', flush=True)
            #print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated(), flush=True)
    
    test_loss /= len(test_loader.dataset)

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)), flush=True)
    return  test_loss, 100. * correct / len(test_loader.dataset)


def run_train_test_loop(model: torch.nn.Module, train_loader, test_loader, model_name: str, epochs: int = 200, device: torch.device = torch.device("cpu")):
    sns.set_style("darkgrid")
#     lr = 1e-3
#     optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr':lr}], lr=lr)
#     scheduler = StepLR(optimizer, step_size=3, gamma=0.7)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay

    criterion = nn.CrossEntropyLoss()
    #print('b1')
    val_best = 0.
    val_accs = []
    train_accs = []
    val_losses = []
    train_losses = []
    train_iters = []
    val_epochs = []
    for epoch in range(1, epochs + 1):
        #print('mid', flush=True)
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = test(model, device, test_loader, criterion)
        #print('b23')
        
        #print('b4')
        scheduler.step(epoch)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_epochs.append(epoch)
        
        clear_output(True)
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        ax1.set_yscale('log')
        lns1_1 = ax1.plot(val_epochs, train_losses, label = "train loss", color = 'green')
        lns1_0 = ax1.plot(val_epochs, val_losses, label = "validation loss", color = 'black')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #color = 'tab:green'
        lns2_0 = ax2.plot(val_epochs, val_accs, label="validation accuracy")
        lns2_1 = ax2.plot(val_epochs, train_accs, label="train accuracy")
        ax2.set_ylabel('accuracy',)  # we already handled the x-label with ax1
        ax2.tick_params(axis = 'y')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        leg = lns1_0 + lns1_1 + lns2_0 + lns2_1
        labs = [l.get_label() for l in leg]
        ax1.legend(leg, labs, loc=0)
        plt.show()

        

        if val_acc > val_best:
            val_best = val_acc
            torch.save(model.state_dict(), f"models/{model_name}.pt")
            
    
    print(f"best acc: {val_best}")
