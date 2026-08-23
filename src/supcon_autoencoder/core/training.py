"""Module for training loop implementation."""

import logging
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch.nn.utils import get_total_norm

from supcon_autoencoder.core.trackers import ExperimentTracker, Phase

from .model import Autoencoder, augment_samples_with_labels

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from supcon_autoencoder.core.checkpoints import Checkpointer, CheckpointState
    from supcon_autoencoder.core.loss import HybridLossItem

    from .data import Sample
    from .loss import HybridLoss
    from .model import AugmentationModule, Autoencoder


logger = logging.getLogger(__name__)


class LossItem(NamedTuple):
    """Loss dictionary for validation."""

    reconstruction_loss: float
    contrastive_loss: float
    hybrid_loss: float


class TrainMetrics(NamedTuple):
    """Metrics collected over a training epoch."""

    reconstruction_loss: float
    contrastive_loss: float
    hybrid_loss: float
    grad_norm_mean: float
    grad_norm_max: float


def _compute_grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Compute the total L2 norm of gradients across the given parameters.

    Args:
        parameters: Model parameters whose gradients are measured.

    Returns:
        float: Total L2 norm of all parameter gradients, or 0.0 if no
            gradients are present.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return get_total_norm(grads).item()


class Trainer:
    """Trainer class for training the autoencoder."""

    def __init__(
        self,
        model: Autoencoder,
        optimizer: Optimizer,
        loss_fn: HybridLoss,
        augmentation_module: AugmentationModule | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Autoencoder model.
            optimizer: PyTorch optimizer.
            loss_fn: Hybrid loss function.
            augmentation_module: Augmentation module for data augmentation.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.augmentation_module = augmentation_module

    def _train_epoch(
        self, loader: DataLoader[Sample], device: torch.device
    ) -> TrainMetrics:
        """Run one training epoch over the dataset.

        Args:
            loader: DataLoader for training data.
            device: Device to load data onto.

        Returns:
            TrainMetrics: Average losses and gradient norm statistics
                over the epoch.
        """
        self.model.train()
        total_supcon_loss = 0.0
        total_recon_loss = 0.0
        total_hybrid_loss = 0.0
        total_grad_norm = 0.0
        max_grad_norm = 0.0
        total_samples = 0
        num_batches = 0
        for batch in loader:
            inputs: torch.Tensor = batch["features"].to(device)
            labels: torch.Tensor = batch["labels"].to(device)

            original_inputs = inputs

            if self.augmentation_module is not None:
                inputs, labels, sample_indices = augment_samples_with_labels(
                    self.augmentation_module, inputs, labels
                )
                original_inputs = original_inputs[
                    sample_indices
                ]  # extend original inputs to match dimensions

            self.optimizer.zero_grad(set_to_none=True)

            embeddings: torch.Tensor = self.model.encoder(inputs)
            reconstructions: torch.Tensor = self.model.decoder(embeddings)

            loss: HybridLossItem = self.loss_fn(
                embeddings=embeddings,
                labels=labels,
                original_input=original_inputs,
                reconstructed_input=reconstructions,
            )

            loss["hybrid_loss"].backward()

            grad_norm = _compute_grad_norm(self.model.parameters())
            total_grad_norm += grad_norm
            max_grad_norm = max(max_grad_norm, grad_norm)
            num_batches += 1

            self.optimizer.step()

            batch_size = inputs.size(0)
            total_supcon_loss += loss["contrastive_loss"] * batch_size
            total_recon_loss += loss["reconstruction_loss"] * batch_size
            total_hybrid_loss += loss["hybrid_loss"].item() * batch_size
            total_samples += batch_size

        return TrainMetrics(
            contrastive_loss=total_supcon_loss / total_samples,
            reconstruction_loss=total_recon_loss / total_samples,
            hybrid_loss=total_hybrid_loss / total_samples,
            grad_norm_mean=total_grad_norm / num_batches,
            grad_norm_max=max_grad_norm,
        )

    def _validate_epoch(
        self, loader: DataLoader[Sample], device: torch.device
    ) -> LossItem:
        """Run one validation epoch over the dataset.

        Args:
            loader: DataLoader for validation data.
            device: Device to load data onto.

        Returns:
            float: Average loss over the epoch.
        """
        self.model.eval()
        total_supcon_loss = 0.0
        total_recon_loss = 0.0
        total_hybrid_loss = 0.0
        total_samples = 0
        with torch.inference_mode():
            for batch in loader:
                inputs: torch.Tensor = batch["features"].to(device)
                labels: torch.Tensor = batch["labels"].to(device)

                embeddings: torch.Tensor = self.model.encoder(inputs)
                reconstructions: torch.Tensor = self.model.decoder(embeddings)

                loss: HybridLossItem = self.loss_fn(
                    embeddings=embeddings,
                    labels=labels,
                    original_input=inputs,
                    reconstructed_input=reconstructions,
                )

                batch_size = inputs.size(0)
                total_supcon_loss += loss["contrastive_loss"] * batch_size
                total_recon_loss += loss["reconstruction_loss"] * batch_size
                total_hybrid_loss += loss["hybrid_loss"].item() * batch_size
                total_samples += batch_size
        return LossItem(
            contrastive_loss=total_supcon_loss / total_samples,
            reconstruction_loss=total_recon_loss / total_samples,
            hybrid_loss=total_hybrid_loss / total_samples,
        )

    def train(  # noqa: PLR0913, PLR0917
        self,
        train_loader: DataLoader[Sample],
        device: torch.device,
        epochs: int,
        val_loader: DataLoader[Sample] | None = None,
        experiment_trackers: list[ExperimentTracker] | None = None,
        checkpointers: list[Checkpointer] | None = None,
        start_epoch: int = 0,
    ) -> None:
        """Run training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            device: Device to load data onto.
            epochs: Number of epochs to train for.
            experiment_trackers: List of experiment trackers to log metrics to.
            checkpointers: List of checkpointers to save training state with
                at the end of each epoch.
            start_epoch: Epoch to start from, e.g. when resuming from a
                checkpoint.
        """
        experiment_trackers = experiment_trackers or []
        checkpointers = checkpointers or []
        for epoch in range(start_epoch, epochs):
            train_metrics = self._train_epoch(train_loader, device)
            for tracker in experiment_trackers:
                tracker.log_metrics(
                    phase=Phase.TRAIN, step=epoch + 1, metrics=train_metrics._asdict()
                )
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, device)
                for tracker in experiment_trackers:
                    tracker.log_metrics(
                        phase=Phase.VAL, step=epoch + 1, metrics=val_loss._asdict()
                    )
            if checkpointers:
                state: CheckpointState = {
                    "epoch": epoch + 1,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "val_metrics": val_loss._asdict() if val_loss is not None else None,
                }
                for checkpointer in checkpointers:
                    checkpointer.save(state)
