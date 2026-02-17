"""Module for training loop implementation."""

import logging
from typing import TYPE_CHECKING, NamedTuple

import torch

from supcon_autoencoder.core.trackers import ExperimentTracker, Phase

from .model import Autoencoder

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from supcon_autoencoder.core.loss import HybridLossItem

    from .data import Sample
    from .loss import HybridLoss
    from .model import Autoencoder


logger = logging.getLogger(__name__)


class LossItem(NamedTuple):
    """Loss dictionary for training."""

    reconstruction_loss: float
    contrastive_loss: float
    hybrid_loss: float


class Trainer:
    """Trainer class for training the autoencoder."""

    def __init__(
        self,
        model: Autoencoder,
        optimizer: Optimizer,
        loss_fn: HybridLoss,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Autoencoder model.
            optimizer: PyTorch optimizer.
            loss_fn: Hybrid loss function.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def _train_epoch(
        self, loader: DataLoader[Sample], device: torch.device
    ) -> LossItem:
        """Run one training epoch over the dataset.

        Args:
            loader: DataLoader for training data.
            device: Device to load data onto.

        Returns:
            float: Average loss over the epoch.
        """
        self.model.train()
        total_supcon_loss = 0.0
        total_recon_loss = 0.0
        total_hybrid_loss = 0.0
        total_samples = 0
        for batch in loader:
            inputs: torch.Tensor = batch["features"].to(device)
            labels: torch.Tensor = batch["labels"].to(device)
            self.optimizer.zero_grad(set_to_none=True)

            embeddings: torch.Tensor = self.model.encoder(inputs)
            reconstructions: torch.Tensor = self.model.decoder(embeddings)

            loss: HybridLossItem = self.loss_fn(
                embeddings=embeddings,
                labels=labels,
                original_input=inputs,
                reconstructed_input=reconstructions,
            )

            loss["hybrid_loss"].backward()
            self.optimizer.step()

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

    def train(
        self,
        train_loader: DataLoader[Sample],
        device: torch.device,
        epochs: int,
        val_loader: DataLoader[Sample] | None = None,
        experiment_trackers: list[ExperimentTracker] | None = None,
    ) -> None:
        """Run training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            device: Device to load data onto.
            epochs: Number of epochs to train for.
            experiment_trackers: List of experiment trackers to log metrics to.
        """
        experiment_trackers = experiment_trackers or []
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, device)
            for tracker in experiment_trackers:
                tracker.log_metrics(
                    phase=Phase.TRAIN, step=epoch + 1, metrics=train_loss._asdict()
                )
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, device)
                for tracker in experiment_trackers:
                    tracker.log_metrics(
                        phase=Phase.VAL, step=epoch + 1, metrics=val_loss._asdict()
                    )
