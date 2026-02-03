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
from matplotlib import pyplot as plt
from utils.embedding_plot import (  # type: ignore[import-not-found]
    analyze_embeddings,
    compute_embeddings,
    compute_projections,
    ground_truth_score,
    load_model,
    projection_plot,
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
    model = load_model(args.model_path, device)

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
        len(training_embeddings),  # type: ignore[arg-type]
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
            len(validation_embeddings),  # type: ignore[arg-type]
        )
        analyze_embeddings(validation_embeddings, "validation", logger)

    # Train k-means and evaluate clustering
    n_clusters = len(torch.unique(training_embeddings.labels))  # type: ignore[attr-defined]
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

    # Compute 2D projections (training only)
    logger.info("Computing 2D projections (PCA, t-SNE, UMAP)...")
    projections = compute_projections(training_embeddings)
    logger.info("2D projections computed")
    training_labels_numeric = training_embeddings.labels.cpu().numpy()  # type: ignore[attr-defined]

    # Convert numeric labels to string labels using label encoder
    label_encoder = LabelEncoder.from_json(data_training_config.label_encoder_file)
    label_map: dict[int, str] = label_encoder.__reversed__()
    training_labels = np.array(
        [label_map[int(label)] for label in training_labels_numeric]
    )

    # Create projection plot
    logger.info("Creating projection plot...")
    fig = projection_plot(
        projections,
        training_labels,
        training_scores,
        validation_scores,
    )
    logger.info("Projection plot created")

    # Save projection plot if output path provided
    if args.projection_output is not None:
        fig.savefig(args.projection_output, dpi=300, bbox_inches="tight")

    # Show projection plot if requested
    if args.show:
        plt.show()
