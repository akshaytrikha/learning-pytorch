from pathlib import Path
import requests
import shutil
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, List


def fetch_data():
    """fetch data if doesn't exist"""
    data_dir = Path("./data")
    zip_path = data_dir / "pizza_steak_sushi.zip"

    if not data_dir.exists():
        print("Downloading data")
        data_dir.mkdir(exist_ok=True)

        # fetch .zip
        with open(zip_path, "wb") as f:
            result = requests.get(
                "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
            )
            f.write(result.content)

        # unpack zipq
        shutil.unpack_archive(zip_path, data_dir, "zip")
        os.remove(zip_path)


def create_dataloaders(
    train_dir: Path,
    dev_dir: Path,
    batch_size: int,
    transform: transforms.Compose,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """create dataloaders from corresponding directories

    Args:
        train_dir (Path): training directory.
        test_dir (Path) to validation directory.
        transform (torchvision.transforms.Compose) to perform on training and testing data.
        batch_size (int) number of samples per batch in each of the DataLoaders.
        num_workers (int) for number of workers per DataLoader.

    Returns:
        Tuple[train_dataloader, test_dataloader, class_names]
    """
    # read data
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    dev_data = datasets.ImageFolder(dev_dir, transform=transform)

    class_names = train_data.classes

    # convert images into dataloaders
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    dev_dataloader = DataLoader(
        dev_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_dataloader, dev_dataloader, class_names
