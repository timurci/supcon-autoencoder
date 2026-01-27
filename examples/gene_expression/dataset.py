"""Module for Gene Expression Dataset."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from examples.gene_expression.config import DataConfig
    from supcon_autoencoder.core.data import Sample


class LabelEncoder:
    """Label encoder for converting labels to numeric values."""

    def __init__(self, label_map: dict[str, int]) -> None:
        """Initialize the label encoder with a label map."""
        values = label_map.values()
        if len(values) != len(set(values)):
            msg = "Label map values must be unique"
            raise ValueError(msg)

        self.label_map = label_map.copy()

    def __reversed__(self) -> dict[int, str]:
        """Return the reversed label map."""
        return {v: k for k, v in self.label_map.items()}

    @staticmethod
    def from_labels(labels: list[str]) -> LabelEncoder:
        """Create a label encoder from a list of labels."""
        return LabelEncoder({label: idx for idx, label in enumerate(labels)})

    def to_json(self, path: str) -> None:
        """Save the label encoder to a json file."""
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.label_map, f)

    @staticmethod
    def from_json(path: str) -> LabelEncoder:
        """Load the label encoder from a json file."""
        with Path(path).open("r", encoding="utf-8") as f:
            label_map = json.load(f)
        return LabelEncoder(label_map)


class LabeledGeneExpressionDataset(Dataset):
    """Dataset representing gene expression data with labels.

    The expression data and metadata are loaded from parquet files. Both files
    must contain a column with the sample IDs. The metadata file must contain a
    column with the labels.
    """

    def __init__(
        self,
        expression_file: str,
        metadata_file: str,
        id_column: str,
        label_column: str,
        label_encoder: LabelEncoder,
    ) -> None:
        """Initialize the dataset with the given parameters.

        Args:
            expression_file: Parquet file path containing the gene expression data.
            metadata_file: Path to the parquet file containing the metadata.
            id_column: Name of the column containing the sample IDs.
            label_column: Name of the column containing the labels.
            label_encoder: Label encoder for converting labels to numeric values.
        """
        exprs: pl.DataFrame = pl.read_parquet(expression_file)
        metadata: pl.DataFrame = pl.read_parquet(metadata_file).select(
            id_column, label_column
        )
        metadata = metadata.select(
            pl.col(id_column),
            pl.col(label_column).replace_strict(label_encoder.label_map),
        )

        data: pl.DataFrame = exprs.join(metadata, on=id_column, how="inner")

        self.sample_ids = data[id_column].to_numpy().copy()
        self.labels = torch.from_numpy(data[label_column].to_numpy().copy())
        self.features = torch.from_numpy(
            data.drop([id_column, label_column]).to_numpy().copy()
        )

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Sample:
        """Get a sample from the dataset."""
        features = self.features[index, :]
        label = self.labels[index]

        return {
            "features": features,
            "labels": label,
        }

    def to(self, device: torch.device) -> None:
        """Move the features and labels to the specified device."""
        self.labels = self.labels.to(device)
        self.features = self.features.to(device)


def create_dataloader(data_config: DataConfig) -> DataLoader[Sample]:
    """Create a dataloader from a data config."""
    dataset = LabeledGeneExpressionDataset(
        expression_file=data_config.expression_file,
        metadata_file=data_config.metadata_file,
        id_column=data_config.id_column,
        label_column=data_config.label_column,
        label_encoder=LabelEncoder.from_json(data_config.label_encoder_file),
    )
    return DataLoader(
        dataset, batch_size=data_config.batch_size, shuffle=data_config.shuffle
    )
