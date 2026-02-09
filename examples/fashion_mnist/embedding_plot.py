"""Embedding visualization script for Fashion-MNIST data.

This is a hardcoded example demonstrating how to compute and visualize embeddings
from a trained SupCon autoencoder on Fashion-MNIST dataset.
"""

import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

# Add parent directory to path for imports when running from examples/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from utils.embedding_plot import (  # type: ignore[import-not-found]
    analyze_embeddings,
    compute_embeddings,
    compute_projections,
    ground_truth_score,
    load_model,
    projection_plot,
    train_kmeans,
)

from .dataset import create_dataloader

# Hardcoded parameters
DATA_ROOT = "./data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optional: Set to None to use all data
RAND_SUBSET_SIZE = None

# Optional: Set to None for non-reproducible results
SEED = 42

# Fashion-MNIST class names for label mapping
FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def build_parser() -> ArgumentParser:
    """Create argument parser."""
    parser = ArgumentParser()
    parser.add_argument(
        "--model-path", required=True, help="Path to the trained model file."
    )
    parser.add_argument(
        "--projection-output", required=True, help="Path to save the projection plot."
    )
    return parser


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run embedding visualization pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    logger.info("Using device: %s", DEVICE)

    # Set random seed
    if SEED is not None:
        torch.manual_seed(SEED)
        logger.info("Random seed set to %d", SEED)

    # Create datasets (data should already be downloaded from training)
    logger.info("Loading Fashion-MNIST datasets from %s...", DATA_ROOT)
    training_dataset = create_dataloader(
        root=DATA_ROOT,
        batch_size=256,
        train=True,
        download=False,
        shuffle=False,
    ).dataset
    validation_dataset = create_dataloader(
        root=DATA_ROOT,
        batch_size=256,
        train=False,
        download=False,
        shuffle=False,
    ).dataset
    logger.info(
        "Datasets loaded - Train: %d samples, Val: %d samples",
        len(training_dataset),  # type: ignore[arg-type]
        len(validation_dataset),  # type: ignore[arg-type]
    )

    # Load trained model and move to device
    model = load_model(args.model_path, DEVICE)
    model = model.to(DEVICE)

    # Compute training embeddings
    logger.info("Computing training embeddings...")
    training_embeddings = compute_embeddings(
        training_dataset,
        model.encoder,
        DEVICE,
        rand_subset_size=RAND_SUBSET_SIZE,
    )
    logger.info(
        "Training embeddings computed: %d samples",
        len(training_embeddings),
    )
    analyze_embeddings(training_embeddings, "training", logger)

    # Compute validation embeddings
    logger.info("Computing validation embeddings...")
    validation_embeddings = compute_embeddings(
        validation_dataset,
        model.encoder,
        DEVICE,
        rand_subset_size=RAND_SUBSET_SIZE,
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
    training_labels_numeric = training_embeddings.labels.cpu().numpy()

    # Map numeric labels to class names for visualization
    training_labels = np.array(
        [FASHION_MNIST_CLASSES[int(label)] for label in training_labels_numeric]
    )

    # Create projection plot
    logger.info("Creating projection plot...")
    fig = projection_plot(
        projections,
        training_labels,
        training_scores,
        validation_scores,
        title="2D Projections of Fashion-MNIST Embeddings",
    )
    logger.info("Projection plot created")

    # Save projection plot
    fig.savefig(args.projection_output, dpi=300, bbox_inches="tight")
    logger.info("Projection plot saved to %s", args.projection_output)

    logger.info("Embedding visualization complete!")


if __name__ == "__main__":
    main()
