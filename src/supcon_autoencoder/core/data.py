"""Module for PyTorch Dataset implementation."""

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import torch


class Sample(TypedDict):
    """Dataset sample interface."""

    features: torch.Tensor
    labels: torch.Tensor
