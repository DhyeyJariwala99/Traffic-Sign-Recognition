import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

IMG_SIZE = 32
BATCH_SIZE = 64
NUM_CLASSES = 43
DATA_DIR = "./data"


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171),
                                 (0.2672, 0.2564, 0.2629)),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171),
                             (0.2672, 0.2564, 0.2629)),
    ])


def get_loaders():
    train_full = datasets.GTSRB(
        root=DATA_DIR, split="train",
        download=True, transform=get_transforms(train=True)
    )
    test_set = datasets.GTSRB(
        root=DATA_DIR, split="test",
        download=True, transform=get_transforms(train=False)
    )

    val_size = int(0.15 * len(train_full))
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader
