from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_transforms(train=True):
    ops = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] if train else []
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def get_dataloaders(data_dir, batch_size=64, num_workers=0):
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=get_transforms(True))
    val = datasets.CIFAR10(data_dir, train=False, download=True, transform=get_transforms(False))
    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    return DataLoader(train, shuffle=True, **kwargs), DataLoader(val, shuffle=False, **kwargs)
