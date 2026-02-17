"""Training script for SupCon autoencoder on gene expression data."""

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml
from dec_torch.autoencoder import AutoEncoder
from torch import nn

from supcon_autoencoder.core.loss import HybridLoss, SupConLoss
from supcon_autoencoder.core.trackers import MLflowTracker, StandardLoggingTracker
from supcon_autoencoder.core.training import Trainer

if TYPE_CHECKING:
    from supcon_autoencoder.core.model import Autoencoder

from .config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingLoopConfig,
)
from .dataset import (
    create_dataloader,
)
from .model import create_autoencoder

logger = logging.getLogger(__name__)


def build_parser() -> ArgumentParser:
    """Create argument parser."""
    parser = ArgumentParser()
    parser.add_argument("--data-config", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--training-config", type=str, required=True)
    parser.add_argument(
        "--model-output", required=True, help="Path to save the trained model."
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
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
) -> Autoencoder:
    """Train a SupCon autoencoder model.

    Args:
        model_config: Model configuration.
        optimizer_config: Optimizer configuration.
        loss_config: Loss configuration.
        training_loop_config: Training loop configuration.
        data_training_config: Training data configuration.
        data_validation_config: Validation data configuration.

    Returns:
        Trained autoencoder model
    """
    training_loader = create_dataloader(data_training_config)
    validation_loader = None
    if data_validation_config is not None:
        validation_loader = create_dataloader(data_validation_config)

    input_dim = training_loader.dataset[0]["features"].shape[0]

    logger.debug("Autoencoder input/output dimension: %d", input_dim)

    model = create_autoencoder(input_dim, model_config=model_config)
    model = model.to(torch.device(training_loop_config.device))

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

    logging_interval = training_loop_config.num_epochs // 10

    params = {
        "training_data": Path(data_training_config.expression_file).name,
        "metadata": Path(data_training_config.metadata_file).name,
        "batch_size": data_training_config.batch_size,
        "latent_dim": model_config.latent_dim,
        "hidden_dims": model_config.hidden_dims,
        "input_dropout": model_config.input_dropout,
        "encoder_activation": model_config.encoder_activation,
        "decoder_activation": model_config.decoder_activation,
        "hidden_activation": model_config.hidden_activation,
        "learning_rate": optimizer_config.learning_rate,
        "optimizer": str(optimizer),
        "supcon_temperature": loss_config.supcon_temperature,
        "hybrid_lambda": loss_config.hybrid_lambda,
        "num_epochs": training_loop_config.num_epochs,
    }

    if data_validation_config is not None:
        params["validation_data"] = Path(data_validation_config.expression_file).name

    with (
        StandardLoggingTracker(
            logger=logger,
            logging_interval=logging_interval,
            experiment_steps=training_loop_config.num_epochs,
        ) as logging_tracker,
        MLflowTracker(
            experiment_name="gene-expression-supcon-autoencoder"
        ) as mlflow_tracker,
    ):
        logging_tracker.log_params(params)
        mlflow_tracker.log_params(params)

        trainer.train(
            train_loader=training_loader,
            device=torch.device(training_loop_config.device),
            epochs=training_loop_config.num_epochs,
            val_loader=validation_loader,
            experiment_trackers=[logging_tracker, mlflow_tracker],
        )

    return model


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    model_yaml = load_yaml(args.model_config)
    training_yaml = load_yaml(args.training_config)
    data_yaml = load_yaml(args.data_config)

    # Load configurations
    model_config = ModelConfig(**model_yaml["model"])
    optimizer_config = OptimizerConfig(**training_yaml["optimizer"])
    loss_config = LossConfig(**training_yaml["loss"])
    training_loop_config = TrainingLoopConfig(**training_yaml["training_loop"])

    data_training_config = DataConfig(**data_yaml["data"]["training"])
    data_validation_config = None
    if data_yaml["data"]["validation"] is not None:
        data_validation_config = DataConfig(**data_yaml["data"]["validation"])

    model = train(
        model_config,
        optimizer_config,
        loss_config,
        training_loop_config,
        data_training_config,
        data_validation_config,
    )

    if isinstance(model, AutoEncoder):
        model.save(args.model_output)
    else:
        torch.save(model.state_dict(), args.model_output)
