"""Module for training checkpointing."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor
    from torch.optim import Optimizer

    from .model import Autoencoder


class CheckpointState(TypedDict):
    """Snapshot of training state at an epoch boundary.

    Attributes:
        epoch: Number of completed epochs (1-based).
        model: Model state dictionary.
        optimizer: Optimizer state dictionary.
        val_metrics: Validation metrics of the completed epoch, or None if
            validation did not run.
    """

    epoch: int
    model: dict[str, Tensor]
    optimizer: dict[str, Any]
    val_metrics: Mapping[str, float] | None


class Checkpointer(Protocol):
    """Interface for saving and loading checkpoints at epoch boundaries."""

    def save(self, state: CheckpointState) -> None:
        """Save a checkpoint of the training state.

        Args:
            state: Snapshot of the training state.
        """
        ...

    def load(
        self,
        model: Autoencoder,
        optimizer: Optimizer,
        device: torch.device | None = None,
        kind: Literal["latest", "best"] = "latest",
    ) -> int:
        """Load a checkpoint into the model and optimizer.

        Args:
            model: Model to restore weights into.
            optimizer: Optimizer to restore state into.
            device: Device to map the checkpoint tensors to. If None, tensors
                are loaded onto their original device.
            kind: Which checkpoint to load, "latest" or "best".

        Returns:
            int: Number of completed epochs recorded in the checkpoint, to be
                passed as ``start_epoch`` to ``Trainer.train``.
        """
        ...


class LocalCheckpointer:
    """Checkpointer that saves latest and best checkpoints to a directory.

    Writes ``latest.pt`` on every save for resuming training, and
    ``best.pt`` whenever the monitored validation metric improves.
    """

    def __init__(
        self,
        directory: str | Path,
        monitor: str = "hybrid_loss",
        mode: Literal["min", "max"] = "min",
    ) -> None:
        """Initialize a local checkpointer.

        Args:
            directory: Directory to write checkpoints to. Created if missing.
            monitor: Validation metric used for best-model selection.
            mode: Whether lower ("min") or higher ("max") monitored values
                are better.
        """
        self.directory = Path(directory)
        self.monitor = monitor
        self.mode = mode
        self._best: float | None = None
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, state: CheckpointState) -> None:
        """Save the checkpoint to disk.

        Always writes ``latest.pt``. Writes ``best.pt`` only when validation
        metrics are present and the monitored metric improves.

        Args:
            state: Snapshot of the training state.
        """
        _atomic_save(state, self.directory / "latest.pt")
        metrics = state["val_metrics"]
        if metrics is None or self.monitor not in metrics:
            return
        value = metrics[self.monitor]
        if self._is_improvement(value):
            self._best = value
            _atomic_save(state, self.directory / "best.pt")

    def _is_improvement(self, value: float) -> bool:
        """Check whether the monitored value improved over the current best.

        Args:
            value: Value of the monitored metric.

        Returns:
            bool: True if the value is the best seen so far.
        """
        if self._best is None:
            return True
        if self.mode == "min":
            return value < self._best
        return value > self._best

    def load(
        self,
        model: Autoencoder,
        optimizer: Optimizer,
        device: torch.device | None = None,
        kind: Literal["latest", "best"] = "latest",
    ) -> int:
        """Load a checkpoint into the model and optimizer.

        Args:
            model: Model to restore weights into.
            optimizer: Optimizer to restore state into.
            device: Device to map the checkpoint tensors to. If None, tensors
                are loaded onto their original device.
            kind: Which checkpoint to load, "latest" or "best".

        Returns:
            int: Number of completed epochs recorded in the checkpoint, to be
                passed as ``start_epoch`` to ``Trainer.train``.

        Raises:
            FileNotFoundError: If the requested checkpoint does not exist.
        """
        state: CheckpointState = torch.load(
            self.directory / f"{kind}.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        return state["epoch"]


def _atomic_save(state: CheckpointState, path: Path) -> None:
    """Write a checkpoint atomically via a temporary file and rename.

    Args:
        state: Checkpoint state to serialize.
        path: Destination path of the checkpoint file.
    """
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)
