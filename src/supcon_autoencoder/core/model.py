"""Module for autoencoder model protocol."""

from typing import TYPE_CHECKING, Protocol, Self

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor, device, nn


class Autoencoder(Protocol):
    """Autoencoder protocol."""

    @property
    def encoder(self) -> nn.Module:
        """Return encoder model."""
        ...

    @property
    def decoder(self) -> nn.Module:
        """Return decoder model."""
        ...

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
