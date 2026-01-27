"""Training script for SupCon autoencoder on gene expression data."""

import json
from pathlib import Path

import torch
import yaml
from dec_torch.autoencoder import AutoEncoder
from torch import nn

from examples.gene_expression.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingLoopConfig,
)
from examples.gene_expression.dataset import (
    create_dataloader,
)
from examples.gene_expression.model import create_autoencoder
from supcon_autoencoder.core.loss import HybridLoss, SupConLoss
from supcon_autoencoder.core.training import EpochLoss, Trainer


def build_parser() -> ArgumentParser:
    """Create argument parser."""
    parser = ArgumentParser()
    parser.add_argument("--data-config", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--training-config", type=str, required=True)
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the trained model."
    )
    return parser


def load_yaml(path: str) -> dict:
    """Load YAML file.

    Args:
        path: Path to YAML file.
    """
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def train(  # noqa: PLR0913
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    loss_config: LossConfig,
    training_loop_config: TrainingLoopConfig,
    data_training_config: DataConfig,
    data_validation_config: DataConfig | None = None,
) -> tuple[nn.Module, list[EpochLoss]]:
    """Train a SupCon autoencoder model.

    Args:
        model_config: Model configuration.
        optimizer_config: Optimizer configuration.
        loss_config: Loss configuration.
        training_loop_config: Training loop configuration.
        data_training_config: Training data configuration.
        data_validation_config: Validation data configuration.
    """
    training_loader = create_dataloader(data_training_config)
    validation_loader = None
    if data_validation_config is not None:
        validation_loader = create_dataloader(data_validation_config)

    input_dim = training_loader.dataset[0]["features"].shape[0]
    model = create_autoencoder(input_dim, model_config=model_config)
    model = model.to(training_loop_config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config.learning_rate)

    loss_fn = HybridLoss(
        sup_con_loss=SupConLoss(temperature=loss_config.supcon_temperature),
        reconstruction_loss=nn.MSELoss(),
        lambda_=loss_config.hybrid_lambda,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )

    history = trainer.train(
        train_loader=training_loader,
        device=torch.device(training_loop_config.device),
        epochs=training_loop_config.num_epochs,
        val_loader=validation_loader,
        logging_interval=training_loop_config.logging_interval,
    )

    return model, history


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = build_parser()
    args = parser.parse_args()

    model_yaml = load_yaml(args.model_config)
    training_yaml = load_yaml(args.training_config)
    data_yaml = load_yaml(args.data_config)

    # Output files
    model_output: str = model_yaml["model"]["output"]
    history_output: str | None = training_yaml["training_loop"].get("history_output")

    # Load configurations
    model_config = ModelConfig(**model_yaml["model"])
    optimizer_config = OptimizerConfig(**training_yaml["optimizer"])
    loss_config = LossConfig(**training_yaml["loss"])
    training_loop_config = TrainingLoopConfig(**training_yaml["training_loop"])

    data_training_config = DataConfig(**data_yaml["data"]["training"])
    data_validation_config = None
    if data_yaml["data"]["validation"] is not None:
        data_validation_config = DataConfig(**data_yaml["data"]["validation"])

    model, history = train(
        model_config,
        optimizer_config,
        loss_config,
        training_loop_config,
        data_training_config,
        data_validation_config,
    )

    if isinstance(model, AutoEncoder):
        model.save(model_output)
    else:
        torch.save(model.state_dict(), model_output)

    if history_output is not None:
        with Path(history_output).open("w") as f:
            json.dump(history, f)
