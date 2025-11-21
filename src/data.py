import random
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


@dataclass
class DataConfig:
    data_dir: str
    batch_size: int = 128
    num_workers: int = 0  # Set to 0 to avoid multiprocessing issues on Windows
    val_split: float = 0.1
    seed: int = 42
    resize_size: int = 224


def _build_transforms(resize_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(resize_size + 32, antialias=True),
            transforms.RandomCrop(resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    return train_transform, eval_transform


def _split_train_val(dataset: datasets.CIFAR10, val_split: float, seed: int):
    if not 0 < val_split < 1:
        raise ValueError("val_split should be between 0 and 1")

    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len], generator=generator)


def get_dataloaders(config: DataConfig) -> Dict[str, DataLoader]:
    train_tf, eval_tf = _build_transforms(config.resize_size)

    full_train = datasets.CIFAR10(
        root=config.data_dir, train=True, transform=train_tf, download=True
    )
    train_subset, val_subset = _split_train_val(full_train, config.val_split, config.seed)

    test_set = datasets.CIFAR10(
        root=config.data_dir, train=False, transform=eval_tf, download=True
    )

    dataloaders = {
        "train": DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        ),
    }

    return dataloaders


def get_class_names(data_dir: str) -> Tuple[str, ...]:
    dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)
    # CIFAR10 targets map to fixed order labels
    return tuple(dataset.classes)
