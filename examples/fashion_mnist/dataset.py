"""Fashion-MNIST dataset for SupCon autoencoder."""

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

if TYPE_CHECKING:
    from supcon_autoencoder.core.data import Sample


class FashionMNISTDataset(Dataset):
    """Fashion-MNIST dataset wrapper for SupCon autoencoder."""

    def __init__(
        self,
        root: str,
        *,
        train: bool = True,
        download: bool = True,
    ) -> None:
        """Initialize Fashion-MNIST dataset.

        Args:
            root: Root directory for dataset.
            train: Whether to load training or test data.
            download: Whether to download the dataset if not present.
        """
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
        )

        self.dataset = datasets.FashionMNIST(
            root=root, train=train, download=download, transform=transform
        )

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        """Get a sample from the dataset."""
        image, label = self.dataset[index]
        return {"features": image, "labels": torch.tensor(label, dtype=torch.long)}


def create_dataloader(
    *,
    root: str = "./data",
    batch_size: int = 256,
    train: bool = True,
    download: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for Fashion-MNIST.

    Args:
        root: Root directory for dataset.
        batch_size: Batch size for the DataLoader.
        train: Whether to load training or test data.
        download: Whether to download the dataset if not present.
        shuffle: Whether to shuffle the data.

    Returns:
        DataLoader for Fashion-MNIST.
    """
    dataset = FashionMNISTDataset(root, train=train, download=download)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
