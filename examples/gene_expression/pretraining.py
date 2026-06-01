"""Pretraining orchestrator for stacked autoencoder."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from dec_torch.autoencoder import StackedAutoEncoder
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def pretrain_phase1(  # noqa: PLR0913
    model: StackedAutoEncoder,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    sae_greedy_epochs: int,
    experiment_trackers: list[Any] | None = None,
) -> None:
    """Run Phase 1: greedy layer-wise pretraining."""
    logger.info("Starting Phase 1: Greedy layer-wise pretraining")
    loss_fn = nn.MSELoss()
    model.greedy_fit(
        train_loader,
        optimizer,
        loss_fn,
        n_epoch=sae_greedy_epochs,
        val_loader=val_loader,
        trackers=experiment_trackers,
    )


def pretrain_phase2(  # noqa: PLR0913
    model: StackedAutoEncoder,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    sae_finetune_epochs: int,
    experiment_trackers: list[Any] | None = None,
) -> None:
    """Run Phase 2: full reconstruction fine-tuning."""
    logger.info("Starting Phase 2: Full reconstruction fine-tuning")
    loss_fn = nn.MSELoss()
    model.fit(
        train_loader,
        optimizer,
        loss_fn,
        n_epoch=sae_finetune_epochs,
        val_loader=val_loader,
        trackers=experiment_trackers,
    )
