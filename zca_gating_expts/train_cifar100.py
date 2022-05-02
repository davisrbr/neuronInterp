import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import torch
#torch.manual_seed(0)
from torchvision import datasets, transforms, models
from gating_resnet import resnet18_sigmoid

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import argparse
import wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--c', default=64, type=int)
    parser.add_argument('--mode', '-mode', default=0, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--aug_typ', default='stan_aug', type=str, choices=['no_aug', 'stan_aug'])
    parser.add_argument('--save', default='no', type=str)
    parser.add_argument('--batch_size', '-b_sz', default=125, type=int)
    args = parser.parse_args()
    best_acc = 0  # best test accuracy
    wandb.init(project="neuronInterp", entity="davisbrownr")
    wandb.config.update(args)


    # Data
    print('==> Preparing data..')
    if args.aug_typ == 'no_aug':
        transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if args.aug_typ == 'stan_aug':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # Model
    print('==> Building model..')
    if args.net == 'resnet18':
        net = models.resnet18(pretrained=False)
        net.fc = torch.nn.Linear(512, 10)
#     if args.net == 'mcnn_linear':
#         net = mCNN_Linear(c=args.c, mode=args.mode)
    net = net.cuda()
    wandb.watch(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)


            if epoch > 0:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()*targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        wandb.log({'tr err': 100.*(1.-correct/total), 'tr loss': train_loss/total}, step = epoch)

    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()*targets.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        wandb.log({'te err': 100.*(1.-correct/total), 'te loss': test_loss/total}, step = epoch)
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc and args.save != 'no':
            print('Saving..')
            torch.save(net.state_dict(), f"models/{args.save}.pt")
            best_acc = acc

        print(acc)
for epoch in range(0, args.epochs):
    train(epoch)
    test(epoch)
    if epoch > 0:
        scheduler.step()
