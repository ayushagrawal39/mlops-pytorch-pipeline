import tarfile
import urllib.request
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
ARCHIVE = "cifar-10-python.tar.gz"
URLS = [
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
    "https://huggingface.co/datasets/zh-plus/tiny-imagenet/resolve/main/cifar-10-python.tar.gz",
]


def get_transforms(train=True):
    ops = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] if train else []
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def _ensure_cifar10(data_dir: str) -> None:
    root = Path(data_dir)
    marker = root / "cifar-10-batches-py" / "data_batch_1"
    if marker.exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    tgz = root / ARCHIVE
    headers = {"User-Agent": "Mozilla/5.0"}
    last = None
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as src, open(tgz, "wb") as dst:
                dst.write(src.read())
            last = None
            break
        except Exception as exc:
            last = exc
    if last is not None and not tgz.exists():
        raise RuntimeError(
            "Could not download CIFAR-10. On the host run:\n"
            f"  curl -L -o {tgz} https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz\n"
            f"  tar -xzf {tgz} -C {root}"
        ) from last
    with tarfile.open(tgz) as tf:
        tf.extractall(root)


def get_dataloaders(data_dir, batch_size=64, num_workers=0):
    _ensure_cifar10(data_dir)
    train = datasets.CIFAR10(data_dir, train=True, download=False, transform=get_transforms(True))
    val = datasets.CIFAR10(data_dir, train=False, download=False, transform=get_transforms(False))
    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    return DataLoader(train, shuffle=True, **kwargs), DataLoader(val, shuffle=False, **kwargs)
