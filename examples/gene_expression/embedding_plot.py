"""Embedding visualization script for gene expression data.

This module provides a CLI wrapper around generic embedding visualization utilities
specifically for gene expression datasets.
"""

import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

# Add parent directory to path for imports when running from examples/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from dec_torch.autoencoder import StackedAutoEncoder
from matplotlib import pyplot as plt
from utils.embedding_plot import (  # type: ignore[import-not-found]
    EmbeddingDataset,
    analyze_embeddings,
    compute_embeddings,
    generate_projection_figure,
    ground_truth_score,
    train_kmeans,
)

from .config import DataConfig
from .dataset import LabeledGeneExpressionDataset, LabelEncoder


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
    parser.add_argument(
        "--projection-output",
        type=str,
        default=None,
        help="Path to save the projection plot. Defaults to None (do not save).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the projection plot interactively.",
    )
    return parser


def _labels_to_names(
    embeddings_dataset: EmbeddingDataset,
    label_map: dict[int, str],
) -> np.ndarray:
    """Convert numeric labels from embeddings to string names using label map.

    Args:
        embeddings_dataset: Dataset with .labels attribute containing numeric labels.
        label_map: Mapping from numeric label to string name.

    Returns:
        Array of string label names.
    """
    labels_numeric = embeddings_dataset.labels.cpu().numpy()
    return np.array([label_map[int(label)] for label in labels_numeric])


if __name__ == "__main__":
    import yaml

    # Parse CLI arguments
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        logger.info("Random seed set to %d", args.seed)

    # Load data configuration
    with Path(args.data_config).open() as f:
        data_yaml = yaml.safe_load(f)

    data_training_config = DataConfig(**data_yaml["data"]["training"])
    data_validation_config = None
    if data_yaml["data"]["validation"] is not None:
        data_validation_config = DataConfig(**data_yaml["data"]["validation"])

    # Create training dataset
    training_dataset = LabeledGeneExpressionDataset(
        expression_file=data_training_config.expression_file,
        metadata_file=data_training_config.metadata_file,
        id_column=data_training_config.id_column,
        label_column=data_training_config.label_column,
        label_encoder=LabelEncoder.from_json(data_training_config.label_encoder_file),
    )

    # Create validation dataset
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

    # Load trained model
    device = torch.device(args.device)
    model = StackedAutoEncoder.load(args.model_path, map_location=device)

    # Compute training embeddings
    logger.info("Computing training embeddings...")
    training_embeddings = compute_embeddings(
        training_dataset,
        model.encoder,
        device,
        rand_subset_size=args.rand_subset_size,
    )
    logger.info(
        "Training embeddings computed: %d samples",
        len(training_embeddings),
    )
    analyze_embeddings(training_embeddings, "training", logger)

    # Compute validation embeddings
    validation_embeddings = None
    if validation_dataset is not None:
        logger.info("Computing validation embeddings...")
        validation_embeddings = compute_embeddings(
            validation_dataset,
            model.encoder,
            device,
            rand_subset_size=args.rand_subset_size,
        )
        logger.info(
            "Validation embeddings computed: %d samples",
            len(validation_embeddings),
        )
        analyze_embeddings(validation_embeddings, "validation", logger)

    # Train k-means and evaluate clustering
    n_clusters = len(torch.unique(training_embeddings.labels))
    logger.info("Fitting K-means model with %d clusters...", n_clusters)
    kmeans_model = train_kmeans(training_embeddings, n_clusters=n_clusters)
    logger.info("K-means model fitted")

    logger.info("Evaluating clustering performance...")
    training_scores = ground_truth_score(kmeans_model, training_embeddings)
    logger.info(
        "Training clustering scores - ARI: %.3f, NMI: %.3f",
        training_scores["ari"],
        training_scores["nmi"],
    )
    validation_scores = None
    if validation_embeddings is not None:
        validation_scores = ground_truth_score(kmeans_model, validation_embeddings)
        logger.info(
            "Validation clustering scores - ARI: %.3f, NMI: %.3f",
            validation_scores["ari"],
            validation_scores["nmi"],
        )

    # Convert numeric labels to string labels using label encoder
    label_encoder = LabelEncoder.from_json(data_training_config.label_encoder_file)
    label_map: dict[int, str] = label_encoder.__reversed__()

    # Generate combined projection figure
    logger.info("Computing 2D projections (PCA, t-SNE, UMAP)...")
    training_labels = _labels_to_names(training_embeddings, label_map)

    validation_labels = None
    if validation_embeddings is not None:
        validation_labels = _labels_to_names(validation_embeddings, label_map)

    fig = generate_projection_figure(
        training_embeddings,
        training_labels,
        training_scores,
        validation_scores,
        validation_embedding_dataset=validation_embeddings,
        validation_labels=validation_labels,
        title="2D Projections of the Embeddings",
    )
    logger.info("Projection figure created")

    # Save projection plot if output path provided
    if args.projection_output is not None:
        fig.savefig(args.projection_output, dpi=300, bbox_inches="tight")
        logger.info("Projection plot saved to %s", args.projection_output)

    # Show projection plot if requested
    if args.show:
        plt.show()
