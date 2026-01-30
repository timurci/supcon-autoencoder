"""Embedding visualization script for gene expression data.

This module provides functions to load a trained autoencoder model and compute
embeddings for gene expression datasets. The embeddings can then be visualized.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from dec_torch.autoencoder import AutoEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from supcon_autoencoder.core.data import Sample

from .config import DataConfig
from .dataset import LabeledGeneExpressionDataset, LabelEncoder

if TYPE_CHECKING:
    from supcon_autoencoder.core.model import Autoencoder

BATCH_SIZE: int = 128


class EmbeddingDataset(Dataset[Sample]):
    """Dataset wrapper for pre-computed embeddings.

    This dataset stores embeddings and their corresponding labels in memory,
    providing a Dataset[Sample] interface for downstream processing.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Initialize the embedding dataset.

        Args:
            embeddings: Tensor of shape (N, embedding_dim) containing computed
                embeddings.
            labels: Tensor of shape (N,) containing label indices.
        """
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of embeddings in the dataset."""
        return len(self.embeddings)

    def __getitem__(self, index: int) -> Sample:
        """Return a sample at the given index.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Sample dictionary with 'features' (embedding) and 'labels' keys.
        """
        return {
            "features": self.embeddings[index],
            "labels": self.labels[index],
        }


def load_model(model_path: str, device: torch.device) -> Autoencoder:
    """Load a trained autoencoder model from disk.

    Args:
        model_path: Path to the saved model file.
        device: Device to map the model to during loading.

    Returns:
        Loaded autoencoder model conforming to the Autoencoder protocol.
    """
    return AutoEncoder.load(model_path, map_location=device)


def compute_embeddings(
    dataset: Dataset[Sample],
    encoder: nn.Module,
    device: torch.device,
    rand_subset_size: int | None = None,
) -> Dataset[Sample]:
    """Compute embeddings for a dataset using the encoder.

    This function computes embeddings by passing features through the encoder
    in evaluation/inference mode. The encoder remains on its device while
    data is moved to the specified device for computation.

    Args:
        dataset: Dataset containing samples with 'features' and 'labels'.
        encoder: Encoder module to use for computing embeddings.
        device: Device to move data tensors to during computation.
        rand_subset_size: If provided, randomly sample this many elements
            from the dataset before computing embeddings. Defaults to None.

    Returns:
        Dataset[Sample] containing the computed embeddings and matched labels.
    """
    encoder.eval()

    if rand_subset_size is not None and rand_subset_size < len(dataset):  # type: ignore[operator]
        indices = torch.randperm(len(dataset))[:rand_subset_size]  # type: ignore[operator]
        dataset = Subset(dataset, indices.tolist())

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    embeddings_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["labels"]

            embeddings = encoder(features)

            embeddings_list.append(embeddings.cpu())
            labels_list.append(labels.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return EmbeddingDataset(embeddings, labels)


def build_parser() -> ArgumentParser:
    """Create argument parser for command-line usage.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = ArgumentParser(
        description="Compute and visualize embeddings from gene expression data."
    )
    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Path to the data configuration YAML file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for computation (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.",
    )
    parser.add_argument(
        "--rand-subset-size",
        type=int,
        default=None,
        help="Randomly sample this many elements from each dataset. "
        "Defaults to None (use all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible subset sampling. Defaults to None.",
    )
    return parser


if __name__ == "__main__":
    import yaml

    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    with Path(args.data_config).open() as f:
        data_yaml = yaml.safe_load(f)

    data_training_config = DataConfig(**data_yaml["data"]["training"])
    data_validation_config = None
    if data_yaml["data"]["validation"] is not None:
        data_validation_config = DataConfig(**data_yaml["data"]["validation"])

    training_dataset = LabeledGeneExpressionDataset(
        expression_file=data_training_config.expression_file,
        metadata_file=data_training_config.metadata_file,
        id_column=data_training_config.id_column,
        label_column=data_training_config.label_column,
        label_encoder=LabelEncoder.from_json(data_training_config.label_encoder_file),
    )

    validation_dataset = None
    if data_validation_config is not None:
        validation_dataset = LabeledGeneExpressionDataset(
            expression_file=data_validation_config.expression_file,
            metadata_file=data_validation_config.metadata_file,
            id_column=data_validation_config.id_column,
            label_column=data_validation_config.label_column,
            label_encoder=LabelEncoder.from_json(
                data_validation_config.label_encoder_file
            ),
        )

    device = torch.device(args.device)
    model = load_model(args.model_path, device)

    training_embeddings = compute_embeddings(
        training_dataset,
        model.encoder,
        device,
        rand_subset_size=args.rand_subset_size,
    )

    validation_embeddings = None
    if validation_dataset is not None:
        validation_embeddings = compute_embeddings(
            validation_dataset,
            model.encoder,
            device,
            rand_subset_size=args.rand_subset_size,
        )

    # Embeddings are now computed and available for plotting
    # training_embeddings and validation_embeddings are Dataset[Sample] instances
