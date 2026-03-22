"""Module for training loop implementation."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from supcon_autoencoder.core.trackers import ExperimentTracker, Phase

from .model import Autoencoder, augment_samples_with_labels

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from .data import Sample
    from .loss import (
        HybridLoss,
        HybridLossItem,
        JointContrastiveHybridLoss,
        JointContrastiveHybridLossItem,
    )
    from .model import AugmentationModule, Autoencoder


logger = logging.getLogger(__name__)


class Trainer(ABC):
    """Abstract trainer class for training an autoencoder."""

    @abstractmethod
    def _train_epoch(
        self, loader: DataLoader[Sample], device: torch.device
    ) -> dict[str, float]:
        """Run one training epoch over the dataset.

        Args:
            loader: DataLoader for training data.
            device: Device to load data onto.

        Returns:
            A dictionary of average loss over the epoch.
        """
        ...

    @abstractmethod
    def _validate_epoch(
        self, loader: DataLoader[Sample], device: torch.device
    ) -> dict[str, float]:
        """Run one validation epoch over the dataset.

        Args:
            loader: DataLoader for validation data.
            device: Device to load data onto.

        Returns:
            A dictionary of average loss over the epoch.
        """
        ...

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
                    phase=Phase.TRAIN, step=epoch + 1, metrics=train_loss
                )
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, device)
                for tracker in experiment_trackers:
                    tracker.log_metrics(
                        phase=Phase.VAL, step=epoch + 1, metrics=val_loss
                    )


class HybridLossTrainer(Trainer):
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
    ) -> dict[str, float]:
        """Run one training epoch over the dataset.

        Args:
            loader: DataLoader for training data.
            device: Device to load data onto.

        Returns:
            A dictionary of average loss over the epoch.
        """
        self.model.train()
        total_supcon_loss = 0.0
        total_recon_loss = 0.0
        total_hybrid_loss = 0.0
        total_samples = 0
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
            self.optimizer.step()

            batch_size = inputs.size(0)
            total_supcon_loss += loss["supcon_loss"] * batch_size
            total_recon_loss += loss["reconstruction_loss"] * batch_size
            total_hybrid_loss += loss["hybrid_loss"].item() * batch_size
            total_samples += batch_size

        return {
            "supcon_loss": total_supcon_loss / total_samples,
            "reconstruction_loss": total_recon_loss / total_samples,
            "hybrid_loss": total_hybrid_loss / total_samples,
        }

    def _validate_epoch(
        self, loader: DataLoader[Sample], device: torch.device
    ) -> dict[str, float]:
        """Run one validation epoch over the dataset.

        Args:
            loader: DataLoader for validation data.
            device: Device to load data onto.

        Returns:
            A dictionary of average loss over the epoch.
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
                total_supcon_loss += loss["supcon_loss"] * batch_size
                total_recon_loss += loss["reconstruction_loss"] * batch_size
                total_hybrid_loss += loss["hybrid_loss"].item() * batch_size
                total_samples += batch_size
        return {
            "supcon_loss": total_supcon_loss / total_samples,
            "reconstruction_loss": total_recon_loss / total_samples,
            "hybrid_loss": total_hybrid_loss / total_samples,
        }


class JointContrastiveHybridLossTrainer(Trainer):
    """Trainer class for training the autoencoder."""

    def __init__(
        self,
        model: Autoencoder,
        optimizer: Optimizer,
        loss_fn: JointContrastiveHybridLoss,
        augmentation_module: AugmentationModule,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Autoencoder model.
            optimizer: PyTorch optimizer.
            loss_fn: Joint contrastive hybrid loss function.
            augmentation_module: Augmentation module for data augmentation.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.augmentation_module = augmentation_module

    def _train_epoch(
        self, loader: DataLoader[Sample], device: torch.device
    ) -> dict[str, float]:
        """Run one training epoch over the dataset.

        Args:
            loader: DataLoader for training data.
            device: Device to load data onto.

        Returns:
            A dictionary of average loss over the epoch.
        """
        self.model.train()
        total_supcon_loss = 0.0
        total_self_supcon_loss = 0.0
        total_recon_loss = 0.0
        total_hybrid_loss = 0.0
        total_samples = 0
        for batch in loader:
            inputs: torch.Tensor = batch["features"].to(device)
            labels: torch.Tensor = batch["labels"].to(device)

            original_inputs = inputs

            inputs, labels, sample_indices = augment_samples_with_labels(
                self.augmentation_module, inputs, labels
            )
            original_inputs = original_inputs[
                sample_indices
            ]  # extend original inputs to match dimensions

            self.optimizer.zero_grad(set_to_none=True)

            embeddings: torch.Tensor = self.model.encoder(inputs)
            reconstructions: torch.Tensor = self.model.decoder(embeddings)

            loss: JointContrastiveHybridLossItem = self.loss_fn(
                embeddings=embeddings,
                labels=labels,
                sample_indices=sample_indices,
                original_input=original_inputs,
                reconstructed_input=reconstructions,
            )

            loss["hybrid_loss"].backward()
            self.optimizer.step()

            total_supcon_loss += loss["supcon_loss"] * len(inputs)
            total_self_supcon_loss += loss["self_supcon_loss"] * len(inputs)
            total_recon_loss += loss["reconstruction_loss"] * len(inputs)
            total_hybrid_loss += loss["hybrid_loss"].item() * len(inputs)
            total_samples += len(inputs)

        return {
            "supcon_loss": total_supcon_loss / total_samples,
            "self_supcon_loss": total_self_supcon_loss / total_samples,
            "reconstruction_loss": total_recon_loss / total_samples,
            "hybrid_loss": total_hybrid_loss / total_samples,
        }

    def _validate_epoch(
        self, loader: DataLoader[Sample], device: torch.device
    ) -> dict[str, float]:
        """Run one validation epoch over the dataset.

        Args:
            loader: DataLoader for validation data.
            device: Device to load data onto.

        Returns:
            A dictionary of average loss over the epoch.
        """
        self.model.eval()
        total_supcon_loss = 0.0
        total_self_supcon_loss = 0.0
        total_recon_loss = 0.0
        total_hybrid_loss = 0.0
        total_samples = 0
        with torch.inference_mode():
            for batch in loader:
                inputs: torch.Tensor = batch["features"].to(device)
                labels: torch.Tensor = batch["labels"].to(device)

                embeddings: torch.Tensor = self.model.encoder(inputs)
                reconstructions: torch.Tensor = self.model.decoder(embeddings)

                loss: JointContrastiveHybridLossItem = self.loss_fn(
                    embeddings=embeddings,
                    labels=labels,
                    sample_indices=torch.arange(len(inputs), device=labels.device),
                    original_input=inputs,
                    reconstructed_input=reconstructions,
                )

                total_supcon_loss += loss["supcon_loss"] * len(inputs)
                total_self_supcon_loss += loss["self_supcon_loss"] * len(inputs)
                total_recon_loss += loss["reconstruction_loss"] * len(inputs)
                total_hybrid_loss += loss["hybrid_loss"].item() * len(inputs)
                total_samples += len(inputs)
        return {
            "supcon_loss": total_supcon_loss / total_samples,
            "self_supcon_loss": total_self_supcon_loss / total_samples,
            "reconstruction_loss": total_recon_loss / total_samples,
            "hybrid_loss": total_hybrid_loss / total_samples,
        }
