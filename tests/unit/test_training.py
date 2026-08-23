"""Unit tests for training module."""

import math
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from supcon_autoencoder.core.data import Sample
from supcon_autoencoder.core.loss import HybridLoss, SupConLoss
from supcon_autoencoder.core.trackers import Phase
from supcon_autoencoder.core.training import Trainer, TrainMetrics, _compute_grad_norm

if TYPE_CHECKING:
    from collections.abc import Mapping


class TinyAutoencoder(nn.Module):
    """Minimal autoencoder satisfying the Autoencoder protocol."""

    def __init__(self, input_dim: int = 6, latent_dim: int = 3) -> None:
        """Initialize tiny autoencoder.

        Args:
            input_dim: Input and reconstruction dimension.
            latent_dim: Latent embedding dimension.
        """
        super().__init__()
        self._encoder = nn.Linear(input_dim, latent_dim)
        self._decoder = nn.Linear(latent_dim, input_dim)

    @property
    def encoder(self) -> nn.Module:
        """Return encoder model."""
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        """Return decoder model."""
        return self._decoder


class SyntheticDataset(Dataset[Sample]):
    """Synthetic in-memory dataset yielding Sample dicts."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Initialize dataset from feature and label tensors.

        Args:
            features: Feature matrix (num_samples, input_dim).
            labels: Label vector (num_samples).
        """
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.labels)

    def __getitem__(self, index: int) -> Sample:
        """Return the sample at the given index.

        Args:
            index: Sample index.

        Returns:
            Sample with features and labels tensors.
        """
        return {"features": self.features[index], "labels": self.labels[index]}


class RecordingTracker:
    """Experiment tracker that records logged data for assertions."""

    def __init__(self) -> None:
        """Initialize recording tracker with empty records."""
        self.metrics: list[tuple[Phase, int, dict[str, float]]] = []
        self.params: list[dict[str, Any]] = []

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record logged parameters.

        Args:
            params: The parameters to record.
        """
        self.params.append(dict(params))

    def log_metrics(
        self, phase: Phase, step: int, metrics: Mapping[str, float]
    ) -> None:
        """Record logged metrics.

        Args:
            phase: The phase of the experiment run.
            step: The step of the experiment run.
            metrics: The metrics to record.
        """
        self.metrics.append((phase, step, dict(metrics)))


def _make_loader(batch_size: int = 4, num_samples: int = 8) -> DataLoader[Sample]:
    """Create a DataLoader over synthetic samples with balanced labels.

    Args:
        batch_size: Number of samples per batch.
        num_samples: Total number of samples in the dataset.

    Returns:
        DataLoader yielding Sample batches with two balanced classes.
    """
    torch.manual_seed(0)
    dataset = SyntheticDataset(
        features=torch.randn(num_samples, 6),
        labels=torch.arange(num_samples) % 2,
    )
    return DataLoader(dataset, batch_size=batch_size)


def _make_trainer() -> Trainer:
    """Create a Trainer over a tiny randomly initialized autoencoder.

    Returns:
        Trainer with SGD optimizer and hybrid SupCon/reconstruction loss.
    """
    torch.manual_seed(0)
    model = TinyAutoencoder()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = HybridLoss(SupConLoss(temperature=0.5), nn.MSELoss(), lambda_=0.5)
    return Trainer(model, optimizer, loss_fn)


class TestComputeGradNorm:
    """Test suite for _compute_grad_norm helper."""

    def test_returns_zero_without_gradients(self) -> None:
        """Parameters without gradients produce a zero norm."""
        layer = nn.Linear(3, 2)
        assert _compute_grad_norm(layer.parameters()) == 0.0

    def test_computes_total_l2_norm(self) -> None:
        """Total norm matches the L2 norm of all concatenated gradients."""
        layer = nn.Linear(2, 2)
        layer.weight.grad = torch.ones(2, 2)
        layer.bias.grad = torch.ones(2)

        expected = math.sqrt(6.0)  # 4 weight + 2 bias entries of magnitude 1
        assert _compute_grad_norm(layer.parameters()) == pytest.approx(expected)

    def test_ignores_parameters_without_gradients(self) -> None:
        """Parameters with None gradients are excluded from the norm."""
        layer = nn.Linear(2, 2)
        layer.weight.grad = torch.ones(2, 2)
        layer.bias.grad = None

        expected = math.sqrt(4.0)
        assert _compute_grad_norm(layer.parameters()) == pytest.approx(expected)


class TestTrainerGradTracking:
    """Test suite for gradient norm tracking in Trainer."""

    def test_train_metrics_include_grad_norms(self) -> None:
        """Training logs grad_norm_mean and grad_norm_max every epoch."""
        trainer = _make_trainer()
        tracker = RecordingTracker()

        trainer.train(
            _make_loader(),
            torch.device("cpu"),
            epochs=2,
            experiment_trackers=[tracker],
        )

        train_metrics = [
            metrics for phase, _, metrics in tracker.metrics if phase == Phase.TRAIN
        ]
        assert len(train_metrics) == 2
        for metrics in train_metrics:
            assert set(metrics) == set(TrainMetrics._fields)
            assert metrics["grad_norm_mean"] > 0.0
            assert metrics["grad_norm_max"] >= metrics["grad_norm_mean"]

    def test_val_metrics_exclude_grad_norms(self) -> None:
        """Validation logs only loss metrics, without gradient statistics."""
        trainer = _make_trainer()
        tracker = RecordingTracker()

        trainer.train(
            _make_loader(),
            torch.device("cpu"),
            epochs=1,
            val_loader=_make_loader(),
            experiment_trackers=[tracker],
        )

        val_metrics = [
            metrics for phase, _, metrics in tracker.metrics if phase == Phase.VAL
        ]
        assert len(val_metrics) == 1
        assert set(val_metrics[0]) == {
            "reconstruction_loss",
            "contrastive_loss",
            "hybrid_loss",
        }

    def test_grad_norm_is_positive_after_backward(self) -> None:
        """Gradient norms are positive for a trainable model with real batches."""
        trainer = _make_trainer()
        metrics = trainer._train_epoch(_make_loader(), torch.device("cpu"))

        assert metrics.grad_norm_mean > 0.0
        assert metrics.grad_norm_max > 0.0
