"""Unit tests for checkpointing module."""

from typing import TYPE_CHECKING, Any, Literal

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from supcon_autoencoder.core.checkpoints import CheckpointState, LocalCheckpointer
from supcon_autoencoder.core.data import Sample
from supcon_autoencoder.core.loss import HybridLoss, SupConLoss
from supcon_autoencoder.core.training import Trainer

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from torch.optim import Optimizer

    from supcon_autoencoder.core.model import Autoencoder
    from supcon_autoencoder.core.trackers import Phase


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
    """Experiment tracker that records logged metrics for assertions."""

    def __init__(self) -> None:
        """Initialize recording tracker with empty records."""
        self.metrics: list[tuple[Phase, int, dict[str, float]]] = []

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record logged parameters.

        Args:
            params: The parameters to record.
        """

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


class RecordingCheckpointer:
    """Checkpointer that records saved states for assertions."""

    def __init__(self) -> None:
        """Initialize recording checkpointer with empty records."""
        self.states: list[CheckpointState] = []

    def save(self, state: CheckpointState) -> None:
        """Record saved checkpoint state.

        Args:
            state: Snapshot of the training state.
        """
        self.states.append(state)

    def load(
        self,
        model: Autoencoder,
        optimizer: Optimizer,
        device: torch.device | None = None,
        kind: Literal["latest", "best"] = "latest",
    ) -> int:
        """Loading is not supported by the recording double."""
        raise NotImplementedError


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
        Trainer with momentum SGD (so optimizer state is populated) and
            hybrid SupCon/reconstruction loss.
    """
    torch.manual_seed(0)
    model = TinyAutoencoder()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = HybridLoss(SupConLoss(temperature=0.5), nn.MSELoss(), lambda_=0.5)
    return Trainer(model, optimizer, loss_fn)


def _make_state(epoch: int = 1, hybrid_loss: float = 1.0) -> CheckpointState:
    """Create a checkpoint state from a fresh tiny autoencoder.

    Args:
        epoch: Number of completed epochs to record.
        hybrid_loss: Value of the hybrid loss in the validation metrics.

    Returns:
        CheckpointState with model/optimizer state and validation metrics.
    """
    model = TinyAutoencoder()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_metrics": {"hybrid_loss": hybrid_loss},
    }


class TestLocalCheckpointer:
    """Test suite for LocalCheckpointer."""

    def test_writes_latest_checkpoint(self, tmp_path: Path) -> None:
        """Every save writes latest.pt with the given state."""
        checkpointer = LocalCheckpointer(tmp_path)
        checkpointer.save(_make_state(epoch=3))

        state: CheckpointState = torch.load(tmp_path / "latest.pt", weights_only=True)
        assert state["epoch"] == 3

    def test_creates_missing_directory(self, tmp_path: Path) -> None:
        """The checkpoint directory is created if it does not exist."""
        directory = tmp_path / "nested" / "checkpoints"
        LocalCheckpointer(directory).save(_make_state())

        assert (directory / "latest.pt").exists()

    def test_best_checkpoint_written_on_improvement(self, tmp_path: Path) -> None:
        """best.pt is written when the monitored metric decreases in min mode."""
        checkpointer = LocalCheckpointer(tmp_path, mode="min")
        checkpointer.save(_make_state(hybrid_loss=1.0))
        checkpointer.save(_make_state(epoch=2, hybrid_loss=0.5))
        checkpointer.save(_make_state(epoch=3, hybrid_loss=0.8))

        state: CheckpointState = torch.load(tmp_path / "best.pt", weights_only=True)
        assert state["epoch"] == 2

    def test_best_checkpoint_max_mode(self, tmp_path: Path) -> None:
        """best.pt tracks the highest monitored value in max mode."""
        checkpointer = LocalCheckpointer(tmp_path, mode="max")
        checkpointer.save(_make_state(hybrid_loss=0.5))
        checkpointer.save(_make_state(epoch=2, hybrid_loss=0.9))
        checkpointer.save(_make_state(epoch=3, hybrid_loss=0.7))

        state: CheckpointState = torch.load(tmp_path / "best.pt", weights_only=True)
        assert state["epoch"] == 2

    def test_no_best_checkpoint_without_val_metrics(self, tmp_path: Path) -> None:
        """best.pt is not written when validation metrics are absent."""
        checkpointer = LocalCheckpointer(tmp_path)
        state = _make_state()
        state["val_metrics"] = None
        checkpointer.save(state)

        assert not (tmp_path / "best.pt").exists()

    def test_no_best_checkpoint_when_monitor_missing(self, tmp_path: Path) -> None:
        """best.pt is not written when the monitored metric is missing."""
        checkpointer = LocalCheckpointer(tmp_path, monitor="unknown_metric")
        checkpointer.save(_make_state())

        assert not (tmp_path / "best.pt").exists()

    def test_no_temporary_files_left(self, tmp_path: Path) -> None:
        """Atomic writes leave no temporary files behind."""
        LocalCheckpointer(tmp_path).save(_make_state())

        assert not list(tmp_path.glob("*.tmp"))


class TestLocalCheckpointerLoad:
    """Test suite for LocalCheckpointer.load."""

    def test_restores_model_weights(self, tmp_path: Path) -> None:
        """Loading a checkpoint restores trained model weights."""
        trainer = _make_trainer()
        trainer.train(_make_loader(), torch.device("cpu"), epochs=2)
        checkpointer = LocalCheckpointer(tmp_path)
        checkpointer.save(
            {
                "epoch": 2,
                "model": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "val_metrics": None,
            }
        )

        fresh = _make_trainer()
        checkpointer.load(fresh.model, fresh.optimizer)

        for trained_param, restored_param in zip(
            trainer.model.parameters(), fresh.model.parameters(), strict=True
        ):
            assert torch.equal(trained_param, restored_param)

    def test_restores_optimizer_state(self, tmp_path: Path) -> None:
        """Loading a checkpoint restores optimizer momentum buffers."""
        trainer = _make_trainer()
        trainer.train(_make_loader(), torch.device("cpu"), epochs=1)
        checkpointer = LocalCheckpointer(tmp_path)
        checkpointer.save(
            {
                "epoch": 1,
                "model": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "val_metrics": None,
            }
        )

        fresh = _make_trainer()
        checkpointer.load(fresh.model, fresh.optimizer)

        original_state = trainer.optimizer.state_dict()["state"]
        restored_state = fresh.optimizer.state_dict()["state"]
        assert len(original_state) > 0
        assert original_state.keys() == restored_state.keys()
        for key in original_state:
            assert torch.equal(
                original_state[key]["momentum_buffer"],
                restored_state[key]["momentum_buffer"],
            )

    def test_returns_completed_epochs(self, tmp_path: Path) -> None:
        """Load returns the epoch count recorded in the checkpoint."""
        checkpointer = LocalCheckpointer(tmp_path)
        checkpointer.save(_make_state(epoch=7))

        trainer = _make_trainer()
        start_epoch = checkpointer.load(trainer.model, trainer.optimizer)

        assert start_epoch == 7

    def test_loads_best_checkpoint(self, tmp_path: Path) -> None:
        """kind="best" restores the snapshot with the best monitored metric."""
        checkpointer = LocalCheckpointer(tmp_path, mode="min")
        checkpointer.save(_make_state(hybrid_loss=1.0))
        checkpointer.save(_make_state(epoch=2, hybrid_loss=0.5))
        checkpointer.save(_make_state(epoch=3, hybrid_loss=0.8))

        trainer = _make_trainer()
        start_epoch = checkpointer.load(trainer.model, trainer.optimizer, kind="best")

        assert start_epoch == 2

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        """Loading a checkpoint that was never saved raises FileNotFoundError."""
        checkpointer = LocalCheckpointer(tmp_path)
        trainer = _make_trainer()

        with pytest.raises(FileNotFoundError):
            checkpointer.load(trainer.model, trainer.optimizer)


class TestTrainerCheckpointing:
    """Test suite for checkpoint integration in Trainer."""

    def test_saves_checkpoint_each_epoch(self) -> None:
        """The trainer passes a state snapshot to checkpointers every epoch."""
        trainer = _make_trainer()
        checkpointer = RecordingCheckpointer()

        trainer.train(
            _make_loader(),
            torch.device("cpu"),
            epochs=3,
            val_loader=_make_loader(),
            checkpointers=[checkpointer],
        )

        assert [state["epoch"] for state in checkpointer.states] == [1, 2, 3]
        for state in checkpointer.states:
            assert state["val_metrics"] is not None
            assert state["model"]
            assert "param_groups" in state["optimizer"]

    def test_checkpoint_without_validation_has_no_val_metrics(self) -> None:
        """Checkpoint states carry no validation metrics without a val loader."""
        trainer = _make_trainer()
        checkpointer = RecordingCheckpointer()

        trainer.train(
            _make_loader(),
            torch.device("cpu"),
            epochs=1,
            checkpointers=[checkpointer],
        )

        assert len(checkpointer.states) == 1
        assert checkpointer.states[0]["val_metrics"] is None

    def test_resume_continues_epoch_numbering(self, tmp_path: Path) -> None:
        """A resumed run continues epoch numbering from the checkpoint."""
        trainer = _make_trainer()
        local_checkpointer = LocalCheckpointer(tmp_path)
        trainer.train(
            _make_loader(),
            torch.device("cpu"),
            epochs=2,
            checkpointers=[local_checkpointer],
        )

        fresh = _make_trainer()
        start_epoch = local_checkpointer.load(fresh.model, fresh.optimizer)

        tracker = RecordingTracker()
        fresh.train(
            _make_loader(),
            torch.device("cpu"),
            epochs=4,
            start_epoch=start_epoch,
            experiment_trackers=[tracker],
        )

        steps = [step for _, step, _ in tracker.metrics]
        assert steps == [3, 4]
