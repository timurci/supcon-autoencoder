"""Tests for the Gene Expression Dataset module."""

import tempfile
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
import torch

from examples.gene_expression.dataset import LabeledGeneExpressionDataset, LabelEncoder

if TYPE_CHECKING:
    from collections.abc import Generator


NUM_GENES = 5
NUM_SAMPLES = 4


@pytest.fixture
def sample_ids() -> list[str]:
    """Fixture providing sample IDs for test data."""
    return [f"sample_{i:03d}" for i in range(NUM_SAMPLES)]


@pytest.fixture
def gene_names() -> list[str]:
    """Fixture providing gene names for expression matrix."""
    return [f"gene_{i}" for i in range(NUM_GENES)]


@pytest.fixture
def expression_data(sample_ids: list[str], gene_names: list[str]) -> pl.DataFrame:
    """Fixture providing synthetic gene expression data.

    Args:
        sample_ids: List of sample identifiers.
        gene_names: List of gene names.

    Returns:
        DataFrame with sample IDs and gene expression values.
    """
    rng = np.random.default_rng(42)
    data: dict[str, list] = {"sample_id": sample_ids}
    for gene in gene_names:
        data[gene] = rng.standard_normal(len(sample_ids)).tolist()
    return pl.DataFrame(data)


@pytest.fixture
def metadata(sample_ids: list[str]) -> pl.DataFrame:
    """Fixture providing sample metadata with labels.

    Args:
        sample_ids: List of sample identifiers.

    Returns:
        DataFrame with sample IDs and labels.
    """
    # Generate alternating labels for all samples
    labels = [f"class_{chr(65 + (i % 2))}" for i in range(NUM_SAMPLES)]
    return pl.DataFrame(
        {
            "sample_id": sample_ids,
            "condition": labels,
            "batch": [f"batch{(i % 2) + 1}" for i in range(NUM_SAMPLES)],
        }
    )


@pytest.fixture
def label_encoder() -> LabelEncoder:
    """Fixture providing a label encoder for test classes."""
    return LabelEncoder({"class_A": 0, "class_B": 1})


@pytest.fixture
def temp_expression_file(expression_data: pl.DataFrame) -> Generator[str]:
    """Fixture creating a temporary parquet file for expression data.

    Args:
        expression_data: Polars DataFrame with expression data.

    Yields:
        Path to temporary parquet file.
    """
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        expression_data.write_parquet(f.name)
        yield f.name


@pytest.fixture
def temp_metadata_file(metadata: pl.DataFrame) -> Generator[str]:
    """Fixture creating a temporary parquet file for metadata.

    Args:
        metadata: Polars DataFrame with metadata.

    Yields:
        Path to temporary parquet file.
    """
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        metadata.write_parquet(f.name)
        yield f.name


class TestLabeledGeneExpressionDatasetInit:
    """Test suite for LabeledGeneExpressionDataset initialization."""

    def test_init_loads_data_correctly(
        self,
        temp_expression_file: str,
        temp_metadata_file: str,
        label_encoder: LabelEncoder,
    ) -> None:
        """Test that __init__ correctly loads and processes data from parquet files.

        Args:
            temp_expression_file: Path to temporary expression parquet file.
            temp_metadata_file: Path to temporary metadata parquet file.
            label_encoder: Label encoder instance.
        """
        dataset = LabeledGeneExpressionDataset(
            expression_file=temp_expression_file,
            metadata_file=temp_metadata_file,
            id_column="sample_id",
            label_column="condition",
            label_encoder=label_encoder,
        )

        assert len(dataset) == NUM_SAMPLES
        assert dataset.features.shape == (NUM_SAMPLES, NUM_GENES)
        assert dataset.labels.shape == (NUM_SAMPLES,)
        assert dataset.sample_ids.shape == (NUM_SAMPLES,)

    def test_init_handles_extra_metadata_columns(
        self,
        temp_expression_file: str,
        temp_metadata_file: str,
        label_encoder: LabelEncoder,
    ) -> None:
        """Test that __init__ ignores extra columns in metadata.

        The metadata file contains an extra 'batch' column which should not
        affect the dataset creation.

        Args:
            temp_expression_file: Path to temporary expression parquet file.
            temp_metadata_file: Path to temporary metadata parquet file
                (with extra columns).
            label_encoder: Label encoder instance.
        """
        dataset = LabeledGeneExpressionDataset(
            expression_file=temp_expression_file,
            metadata_file=temp_metadata_file,
            id_column="sample_id",
            label_column="condition",
            label_encoder=label_encoder,
        )

        assert dataset.features.shape[1] == NUM_GENES

    def test_init_performs_inner_join(
        self,
        expression_data: pl.DataFrame,
        temp_metadata_file: str,
        label_encoder: LabelEncoder,
        gene_names: list[str],
    ) -> None:
        """Test that __init__ performs inner join on sample IDs.

        When expression data has samples not present in metadata,
        those samples should be excluded.

        Args:
            expression_data: Original expression DataFrame.
            temp_metadata_file: Path to temporary metadata parquet file.
            label_encoder: Label encoder instance.
            gene_names: List of gene names.
        """
        # Create expression file with extra sample not in metadata
        new_sample_id = f"sample_{NUM_SAMPLES + 999:03d}"
        extra_row_data = {"sample_id": [new_sample_id]}
        for gene in gene_names:
            extra_row_data[gene] = [1.0]

        extra_data = expression_data.vstack(pl.DataFrame(extra_row_data))

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            extra_data.write_parquet(f.name)
            temp_expr_file = f.name

        dataset = LabeledGeneExpressionDataset(
            expression_file=temp_expr_file,
            metadata_file=temp_metadata_file,
            id_column="sample_id",
            label_column="condition",
            label_encoder=label_encoder,
        )

        assert len(dataset) == NUM_SAMPLES


class TestLabeledGeneExpressionDatasetGetItem:
    """Test suite for LabeledGeneExpressionDataset __getitem__ method."""

    def test_getitem_returns_correct_structure(
        self,
        temp_expression_file: str,
        temp_metadata_file: str,
        label_encoder: LabelEncoder,
    ) -> None:
        """Test that __getitem__ returns sample with correct structure.

        Args:
            temp_expression_file: Path to temporary expression parquet file.
            temp_metadata_file: Path to temporary metadata parquet file.
            label_encoder: Label encoder instance.
        """
        dataset = LabeledGeneExpressionDataset(
            expression_file=temp_expression_file,
            metadata_file=temp_metadata_file,
            id_column="sample_id",
            label_column="condition",
            label_encoder=label_encoder,
        )

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "features" in sample
        assert "labels" in sample

    def test_getitem_returns_tensors(
        self,
        temp_expression_file: str,
        temp_metadata_file: str,
        label_encoder: LabelEncoder,
    ) -> None:
        """Test that __getitem__ returns torch tensors.

        Args:
            temp_expression_file: Path to temporary expression parquet file.
            temp_metadata_file: Path to temporary metadata parquet file.
            label_encoder: Label encoder instance.
        """
        dataset = LabeledGeneExpressionDataset(
            expression_file=temp_expression_file,
            metadata_file=temp_metadata_file,
            id_column="sample_id",
            label_column="condition",
            label_encoder=label_encoder,
        )

        sample = dataset[0]

        assert isinstance(sample["features"], torch.Tensor)
        assert isinstance(sample["labels"], torch.Tensor)

    def test_getitem_returns_correct_values(
        self,
        temp_expression_file: str,
        temp_metadata_file: str,
        label_encoder: LabelEncoder,
    ) -> None:
        """Test that __getitem__ returns correct feature values and encoded labels.

        Args:
            temp_expression_file: Path to temporary expression parquet file.
            temp_metadata_file: Path to temporary metadata parquet file.
            label_encoder: Label encoder instance.
        """
        dataset = LabeledGeneExpressionDataset(
            expression_file=temp_expression_file,
            metadata_file=temp_metadata_file,
            id_column="sample_id",
            label_column="condition",
            label_encoder=label_encoder,
        )

        sample0 = dataset[0]
        # First sample should be class_A (encoded as 0)
        assert sample0["labels"].item() == 0
        # Features should be 1D tensor with NUM_GENES elements
        assert sample0["features"].shape == (NUM_GENES,)

        sample1 = dataset[1]
        # Second sample should be class_B (encoded as 1)
        assert sample1["labels"].item() == 1
        assert sample1["features"].shape == (NUM_GENES,)

    def test_getitem_index_out_of_range(
        self,
        temp_expression_file: str,
        temp_metadata_file: str,
        label_encoder: LabelEncoder,
    ) -> None:
        """Test that __getitem__ raises IndexError for invalid indices.

        Args:
            temp_expression_file: Path to temporary expression parquet file.
            temp_metadata_file: Path to temporary metadata parquet file.
            label_encoder: Label encoder instance.
        """
        dataset = LabeledGeneExpressionDataset(
            expression_file=temp_expression_file,
            metadata_file=temp_metadata_file,
            id_column="sample_id",
            label_column="condition",
            label_encoder=label_encoder,
        )

        with pytest.raises(IndexError):
            _ = dataset[NUM_SAMPLES + 100]  # Only NUM_SAMPLES samples exist


class TestLabeledGeneExpressionDatasetTo:
    """Test suite for LabeledGeneExpressionDataset to method."""

    def test_to_moves_tensors_to_cpu(
        self,
        temp_expression_file: str,
        temp_metadata_file: str,
        label_encoder: LabelEncoder,
    ) -> None:
        """Test that to method moves features and labels to specified device.

        Args:
            temp_expression_file: Path to temporary expression parquet file.
            temp_metadata_file: Path to temporary metadata parquet file.
            label_encoder: Label encoder instance.
        """
        dataset = LabeledGeneExpressionDataset(
            expression_file=temp_expression_file,
            metadata_file=temp_metadata_file,
            id_column="sample_id",
            label_column="condition",
            label_encoder=label_encoder,
        )

        device = torch.device("cpu")
        dataset.to(device)

        assert dataset.features.device == device
        assert dataset.labels.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_moves_tensors_to_cuda(
        self,
        temp_expression_file: str,
        temp_metadata_file: str,
        label_encoder: LabelEncoder,
    ) -> None:
        """Test that to method moves features and labels to CUDA device if available.

        Args:
            temp_expression_file: Path to temporary expression parquet file.
            temp_metadata_file: Path to temporary metadata parquet file.
            label_encoder: Label encoder instance.
        """
        dataset = LabeledGeneExpressionDataset(
            expression_file=temp_expression_file,
            metadata_file=temp_metadata_file,
            id_column="sample_id",
            label_column="condition",
            label_encoder=label_encoder,
        )

        device = torch.device("cuda:0")
        dataset.to(device)

        assert dataset.features.device == device
        assert dataset.labels.device == device

    def test_to_affects_getitem_output(
        self,
        temp_expression_file: str,
        temp_metadata_file: str,
        label_encoder: LabelEncoder,
    ) -> None:
        """Test that after to method, __getitem__ returns tensors on new device.

        Args:
            temp_expression_file: Path to temporary expression parquet file.
            temp_metadata_file: Path to temporary metadata parquet file.
            label_encoder: Label encoder instance.
        """
        dataset = LabeledGeneExpressionDataset(
            expression_file=temp_expression_file,
            metadata_file=temp_metadata_file,
            id_column="sample_id",
            label_column="condition",
            label_encoder=label_encoder,
        )

        device = torch.device("cpu")
        dataset.to(device)

        sample = dataset[0]

        assert sample["features"].device == device
        assert sample["labels"].device == device
