"""Embedding visualization script for gene expression data.

This module provides functions to load a trained autoencoder model and compute
embeddings for gene expression datasets. The embeddings can then be visualized.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import seaborn as sns
import torch
from dec_torch.autoencoder import AutoEncoder
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from umap import UMAP

from supcon_autoencoder.core.data import Sample

from .config import DataConfig
from .dataset import LabeledGeneExpressionDataset, LabelEncoder

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from supcon_autoencoder.core.model import Autoencoder

BATCH_SIZE: int = 128

# Thresholds for embedding variance analysis
EMBEDDING_VARIANCE_ZERO: float = 1e-6
EMBEDDING_VARIANCE_LOW: float = 0.01


class ClusteringScores(TypedDict):
    """Clustering evaluation scores against ground truth labels."""

    ari: float
    """Adjusted Rand Index score."""

    nmi: float
    """Normalized Mutual Information score."""


class Projections(TypedDict):
    """2D projections of embeddings via different dimensionality reduction methods."""

    pca: np.ndarray
    """PCA projection, shape (n_samples, 2)."""

    tsne: np.ndarray
    """t-SNE projection, shape (n_samples, 2)."""

    umap: np.ndarray
    """UMAP projection, shape (n_samples, 2)."""


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


def train_kmeans(
    dataset: Dataset[Sample],
    n_clusters: int,
) -> KMeans:
    """Train a k-means clustering model on the dataset embeddings.

    Args:
        dataset: Dataset containing embeddings to cluster.
        n_clusters: Number of clusters to find.

    Returns:
        Trained KMeans model fitted on the dataset embeddings.
    """
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    embeddings_list = [batch["features"] for batch in dataloader]

    embeddings = torch.cat(embeddings_list, dim=0).cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)

    return kmeans


def ground_truth_score(
    kmeans_model: KMeans,
    dataset: Dataset[Sample],
) -> ClusteringScores:
    """Compute clustering performance scores against ground truth labels.

    Args:
        kmeans_model: Trained KMeans model for prediction.
        dataset: Dataset with embeddings and true labels.

    Returns:
        ClusteringScores containing ARI and NMI metrics.
    """
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    embeddings_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for batch in dataloader:
        embeddings_list.append(batch["features"])
        labels_list.append(batch["labels"])

    embeddings = torch.cat(embeddings_list, dim=0).cpu().numpy()
    true_labels = torch.cat(labels_list, dim=0).cpu().numpy()

    predicted_labels = kmeans_model.predict(embeddings)

    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    return ClusteringScores(ari=ari, nmi=nmi)


def compute_projections(dataset: Dataset[Sample]) -> Projections:
    """Compute 2D projections of embeddings using PCA, t-SNE, and UMAP.

    Args:
        dataset: Dataset containing embeddings to project.

    Returns:
        Projections containing PCA, t-SNE, and UMAP 2D projections.
    """
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    embeddings_list = [batch["features"] for batch in dataloader]
    embeddings = torch.cat(embeddings_list, dim=0).cpu().numpy()

    # PCA projection
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(embeddings)

    # t-SNE projection
    tsne = TSNE(n_components=2)
    tsne_proj = tsne.fit_transform(embeddings)

    # UMAP projection
    umap = UMAP(n_components=2)
    umap_proj = umap.fit_transform(embeddings)

    return Projections(pca=pca_proj, tsne=tsne_proj, umap=umap_proj)


def projection_plot(  # noqa: PLR0913
    projections: Projections,
    labels: np.ndarray,
    training_scores: ClusteringScores,
    validation_scores: ClusteringScores | None = None,
    title: str = "2D Projections of the Embeddings",
    figsize: tuple[int, int] = (18, 6),
) -> Figure:
    """Create a figure with three 2D projection plots.

    Args:
        projections: Projections containing PCA, t-SNE, and UMAP projections.
        labels: Array of true labels for hue coloring.
        training_scores: Clustering scores for training set.
        validation_scores: Clustering scores for validation set (if available).
        title: Main title for the figure. Defaults to
            "2D Projections of the Embeddings".
        figsize: Figure size as (width, height) in inches. Defaults to (18, 6).

    Returns:
        The Figure object containing the three projection subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    methods = [
        ("PCA", projections["pca"]),
        ("t-SNE", projections["tsne"]),
        ("UMAP", projections["umap"]),
    ]

    for ax, (method_name, proj) in zip(axes, methods, strict=True):
        sns.scatterplot(
            x=proj[:, 0],
            y=proj[:, 1],
            hue=labels,
            ax=ax,
            palette="tab10",
            legend="full",
            alpha=0.7,
        )
        ax.set_title(f"{method_name} Projection", fontsize=12, fontweight="bold")
        ax.set_xlabel("Component 1", fontsize=10)
        ax.set_ylabel("Component 2", fontsize=10)
        ax.grid(visible=True, linestyle="--", alpha=0.3)

    # Build score text
    score_text = (
        f"Training ARI: {training_scores['ari']:.3f}, NMI: {training_scores['nmi']:.3f}"
    )
    if validation_scores is not None:
        score_text += (
            f" | Validation ARI: {validation_scores['ari']:.3f}, "
            f"NMI: {validation_scores['nmi']:.3f}"
        )

    fig.suptitle(f"{title}\n{score_text}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    return fig


def analyze_embeddings(
    dataset: Dataset[Sample],
    dataset_name: str,
    logger: logging.Logger,
) -> None:
    """Analyze embedding statistics and log warnings if suspicious patterns detected.

    Args:
        dataset: Embedding dataset to analyze.
        dataset_name: Name of dataset for logging (e.g., "training", "validation").
        logger: Logger instance for output.
    """
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    embeddings_list = [batch["features"] for batch in dataloader]
    embeddings = torch.cat(embeddings_list, dim=0)

    min_val = embeddings.min().item()
    max_val = embeddings.max().item()
    mean_val = embeddings.mean().item()
    std_val = embeddings.std().item()
    n_samples = len(embeddings)
    embedding_dim = embeddings.shape[1]

    logger.info(
        "%s embeddings stats - samples: %d, dim: %d, "
        "min: %.4f, max: %.4f, mean: %.4f, std: %.4f",
        dataset_name,
        n_samples,
        embedding_dim,
        min_val,
        max_val,
        mean_val,
        std_val,
    )

    if std_val < EMBEDDING_VARIANCE_ZERO:
        logger.warning(
            "%s embeddings have near-zero variance (std=%.6f)! "
            "Encoder may be untrained or collapsed.",
            dataset_name,
            std_val,
        )
    elif std_val < EMBEDDING_VARIANCE_LOW:
        logger.warning(
            "%s embeddings have very low variance (std=%.4f). "
            "Consider checking model training.",
            dataset_name,
            std_val,
        )

    if torch.allclose(
        embeddings, torch.zeros_like(embeddings), atol=EMBEDDING_VARIANCE_ZERO
    ):
        logger.error(
            "%s embeddings are all zeros! Model is completely untrained.",
            dataset_name,
        )


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
