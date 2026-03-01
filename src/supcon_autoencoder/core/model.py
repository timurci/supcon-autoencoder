"""Module for autoencoder model protocol."""

from typing import TYPE_CHECKING, Protocol, Self, TypedDict

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor, device, nn


class TorchModule(Protocol):
    """Protocol for PyTorch modules."""

    def train(self) -> Self:
        """Switch to training mode."""
        ...

    def eval(self) -> Self:
        """Switch to evaluation mode."""
        ...

    def to(self, device: device) -> Self:
        """Move parameters to device."""
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        """Return parameters."""
        ...

    def state_dict(self) -> dict[str, Tensor]:
        """Return state dict."""
        ...


class AugmentationResult(TypedDict):
    """Result of augmentation.

    Attributes:
        output: Augmented output tensor.
        sample_indices: Indices of the corresponding original samples.
    """

    outputs: Tensor
    sample_indices: Tensor


class AugmentationModule(TorchModule, Protocol):
    """Protocol for augmentation modules."""

    def __call__(self, x: Tensor) -> AugmentationResult:
        """Apply augmentation to input.

        Args:
            x: Input tensor.

        Returns:
            AugmentationResult: Augmented output and sample labels.
        """
        ...


class Autoencoder(TorchModule, Protocol):
    """Autoencoder protocol."""

    @property
    def encoder(self) -> nn.Module:
        """Return encoder model."""
        ...

    @property
    def decoder(self) -> nn.Module:
        """Return decoder model."""
        ...


def augment_samples_with_labels(
    augmentation_module: AugmentationModule,
    inputs: Tensor,
    labels: Tensor,
) -> tuple[Tensor, Tensor]:
    """Augment samples using the given augmentation module.

    Args:
        augmentation_module: Augmentation module.
        inputs: Input tensor (batch_size, ...).
        labels: Label tensor (batch_size).

    Returns:
        Augmented inputs and labels.
        inputs (batch_size * n_augmentations, ...),
        labels (batch_size * n_augmentations).
    """
    with torch.no_grad():
        augmentation_results = augmentation_module(inputs)
        augmented_inputs = augmentation_results["outputs"]
        sample_labels = augmentation_results["sample_indices"]
        augmented_labels = torch.tensor(
            [labels[idx] for idx in sample_labels],
            dtype=labels.dtype,
            device=labels.device,
        )

    return augmented_inputs, augmented_labels
