"""Configuration data classes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingLoopConfig:
    """Configuration for training loop."""

    num_epochs: int = 1000
    logging_interval: int = 100
    device: str = "cuda"


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for the optimizer."""

    learning_rate: float


@dataclass(frozen=True)
class LossConfig:
    """Configuration for the loss function."""

    supcon_temperature: float
    hybrid_lambda: float


@dataclass(frozen=True)
class DataConfig:
    """Configuration for the data."""

    batch_size: int
    shuffle: bool = False
    expression_file: str
    metadata_file: str
    label_encoder_file: str
    id_column: str
    label_column: str


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the model."""

    input_dim: int
    latent_dim: int
    hidden_dims: list[int] | None = None
    input_dropout: float = 0.3
    decoder_activation: str = "linear"
    encoder_activation: str = "sigmoid"
    hidden_activation: str = "relu"
