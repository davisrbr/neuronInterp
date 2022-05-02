from typing import List

import torch as ch
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    RandomHorizontalFlip,
    Cutout,
    RandomTranslate,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
import sys

sys.path.append("/people/brow843/neuronInterp")
sys.path.append("/people/brow843/neuronInterp/zca_gating_expts")
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm

from torchvision import datasets, transforms, models
from gating_resnet import resnet18_sigmoid, resnet18_fixed
from gating_resnet_cifar import resnet20_sigmoid, resnet20_fixed, bn_model_handler
import wandb

import argparse
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--mode', '-mode', default=0, type=int)
    args = parser.parse_args()

    datasets = {
        "train": torchvision.datasets.CIFAR10("/tmp", train=True, download=True),
        "test": torchvision.datasets.CIFAR10("/tmp", train=False, download=True),
    }

    for (name, ds) in datasets.items():
        print(name)
        writer = DatasetWriter(
            f"/tmp/cifar_{name}.beton", {"image": RGBImageField(), "label": IntField()}
        )
        writer.from_indexed_dataset(ds)

# Note that statistics are wrt to uin8 range, [0,255].
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]

    # BATCH_SIZE = 512
    BATCH_SIZE = 1024

    loaders = {}
    for name in ["train", "test"]:
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice("cuda:0"),
            Squeeze(),
        ]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == "train":
            image_pipeline.extend(
                [
                    RandomHorizontalFlip(),
                    RandomTranslate(padding=2),
                    Cutout(
                        8, tuple(map(int, CIFAR_MEAN))
                    ),  # Note Cutout is done before normalization.
                ]
            )
        image_pipeline.extend(
            [
                ToTensor(),
                ToDevice("cuda:0", non_blocking=True),
                ToTorchImage(),
                Convert(ch.float16),
                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

        # Create loaders
        loaders[name] = Loader(
            f"/tmp/cifar_{name}.beton",
            batch_size=BATCH_SIZE,
            num_workers=8,
            order=OrderOption.RANDOM,
            drop_last=(name == "train"),
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )

        # model = mCNN_bn_k(c=128, num_classes=100).to(memory_format=ch.channels_last).cuda() # model.to(memory_format=ch.channels_last).cuda()
        # model = models.resnet18(pretrained=False)
        # model.fc = ch.nn.Linear(512, 10)
        # model = resnet18_sigmoid(num_classes=10, mode=args.mode)
        model = resnet20_fixed(num_classes=10, mode=args.mode)
        model = model.to(memory_format=ch.channels_last).cuda()
        # EPOCHS = 2048
        # EPOCHS = 1024
        EPOCHS = 500
        lr = 1.5
        momentum = 0.9
        weight_decay = 1e-4

        opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        iters_per_epoch = 50000 // BATCH_SIZE
        lr_schedule = np.interp(
            np.arange((EPOCHS + 1) * iters_per_epoch),
            [0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
            [0, 1, 0],
        )
        scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
        scaler = GradScaler()
        # loss_fn = CrossEntropyLoss(label_smoothing=0.1)
        loss_fn = CrossEntropyLoss()

    wandb.init(
        project="neuronInterp",
        entity="davisbrownr",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        },
    )
    wandb.run.name = f"resnet20_cifar10_fixed_m{args.mode}_ffcv1"
    wandb.config.update(args)
    wandb.watch(model)

    for ep in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for ims, labs in tqdm(loaders["train"]):
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

                # print(loss.item())
                # print(labs.size(0))
                # print(labs.shape)
                train_loss += loss.item() * labs.size(0)
                _, predicted = out.max(1)
                total += labs.size(0)
                correct += predicted.eq(labs).sum().item()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            wandb.log(
                {
                    "train err": 100.0 * (1.0 - correct / total),
                    "train loss": train_loss / total,
                },
                step=ep,
            )


        model.eval()
        with ch.no_grad():
            total_correct, total_num = 0.0, 0.0
            test_loss = 0
            correct = 0
            total = 0
            for ims, labs in tqdm(loaders["test"]):
                with autocast():
                    out = (model(ims) + model(ch.fliplr(ims))) / 2.0  # Test-time augmentation
                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]

                    loss = loss_fn(out, labs)
                    test_loss += loss.item() * labs.size(0)

                wandb.log(
                    {
                        "test err": 100.0 * (1.0 - total_correct / total_num),
                        "test loss": test_loss / total_num,
                    },
                    step=ep,
                )

            print(f"Accuracy: {total_correct / total_num * 100:.1f}%")
