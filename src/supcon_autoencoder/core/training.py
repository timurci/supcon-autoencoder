"""Module for training loop implementation."""

import logging
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import torch

from .model import Autoencoder

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from .data import Sample
    from .loss import HybridLoss
    from .model import Autoencoder


logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Training phase enum.

    Used to distinguish between training and validation phases in the training loop.
    """

    TRAINING = "training"
    VALIDATION = "validation"


class EpochLoss(NamedTuple):
    """Average loss over an epoch."""

    phase: TrainingPhase
    epoch: int
    loss: float


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

    def train_epoch(self, loader: DataLoader[Sample], device: torch.device) -> float:
        """Run one training epoch over the dataset.

        Args:
            loader: DataLoader for training data.
            device: Device to load data onto.

        Returns:
            float: Average loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            inputs: torch.Tensor = batch["feature"].to(device)
            labels: torch.Tensor = batch["label"].to(device)
            self.optimizer.zero_grad(set_to_none=True)

            embeddings: torch.Tensor = self.model.encoder(inputs)
            reconstructions: torch.Tensor = self.model.decoder(embeddings)

            loss: torch.Tensor = self.loss_fn(
                embeddings=embeddings,
                labels=labels,
                original_input=inputs,
                reconstructed_input=reconstructions,
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(loader)

    def validate_epoch(self, loader: DataLoader[Sample], device: torch.device) -> float:
        """Run one validation epoch over the dataset.

        Args:
            loader: DataLoader for validation data.
            device: Device to load data onto.

        Returns:
            float: Average loss over the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.inference_mode():
            for batch in loader:
                inputs: torch.Tensor = batch["feature"].to(device)
                labels: torch.Tensor = batch["label"].to(device)

                embeddings: torch.Tensor = self.model.encoder(inputs)
                reconstructions: torch.Tensor = self.model.decoder(embeddings)

                loss: torch.Tensor = self.loss_fn(
                    embeddings=embeddings,
                    labels=labels,
                    original_input=inputs,
                    reconstructed_input=reconstructions,
                )

                total_loss += loss.item()
        return total_loss / len(loader)

    def train(
        self,
        train_loader: DataLoader[Sample],
        device: torch.device,
        epochs: int,
        val_loader: DataLoader[Sample] | None = None,
        logging_interval: int = 100,
    ) -> list[EpochLoss]:
        """Run training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            device: Device to load data onto.
            epochs: Number of epochs to train for.
            logging_interval: Number of batches between logging.
        """
        history: list[EpochLoss] = []
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, device)
            history.append(
                EpochLoss(phase=TrainingPhase.TRAINING, epoch=epoch, loss=train_loss)
            )
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader, device)
                history.append(
                    EpochLoss(
                        phase=TrainingPhase.VALIDATION, epoch=epoch, loss=val_loss
                    )
                )
            if epoch % logging_interval == 0:
                logger.info(
                    "Training loss %.4f, validation loss %.4f", train_loss, val_loss
                )
        return history
