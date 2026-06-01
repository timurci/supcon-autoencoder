"""Training script for SupCon autoencoder on gene expression data."""

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml
from dec_torch.autoencoder import AutoEncoder, StackedAutoEncoder
from mlflow.utils.name_utils import _generate_random_name
from torch import nn

from supcon_autoencoder.core.loss import HybridLoss, SupConLoss
from supcon_autoencoder.core.trackers import (
    ExperimentTracker,
    MLflowTracker,
    StandardLoggingTracker,
)
from supcon_autoencoder.core.training import Trainer

if TYPE_CHECKING:
    from supcon_autoencoder.core.model import Autoencoder

from .augmentation import GeneExpressionAugmentation
from .config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingLoopConfig,
)
from .dataset import (
    create_dataloader,
    create_tensor_dataloader,
)
from .model import create_stacked_autoencoder
from .pretraining import pretrain_phase1, pretrain_phase2

logger = logging.getLogger(__name__)


def build_parser() -> ArgumentParser:
    """Create argument parser."""
    parser = ArgumentParser()
    parser.add_argument("--data-config", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--training-config", type=str, required=True)
    parser.add_argument(
        "--model-output", required=True, help="Path to save the Phase 3 trained model."
    )
    parser.add_argument(
        "--phase2-model-output",
        default=None,
        help="Path to save the Phase 2 reconstruction-finetuned model.",
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


def _build_params(  # noqa: PLR0913
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    loss_config: LossConfig,
    training_loop_config: TrainingLoopConfig,
    data_training_config: DataConfig,
    data_validation_config: DataConfig | None,
) -> dict[str, object]:
    """Build the common parameter dictionary for all phases."""
    params: dict[str, object] = {
        "training_data": Path(data_training_config.expression_file).name,
        "metadata": Path(data_training_config.metadata_file).name,
        "augmentation": "gaussian",
        "batch_size": data_training_config.batch_size,
        "latent_dim": model_config.latent_dim,
        "hidden_dims": model_config.hidden_dims,
        "input_dropout": model_config.input_dropout,
        "encoder_activation": model_config.encoder_activation,
        "decoder_activation": model_config.decoder_activation,
        "hidden_activation": model_config.hidden_activation,
        "learning_rate": optimizer_config.learning_rate,
        "supcon_temperature": loss_config.supcon_temperature,
        "hybrid_lambda": loss_config.hybrid_lambda,
        "supcon_hybrid_epochs": training_loop_config.supcon_hybrid_epochs,
        "sae_greedy_epochs": training_loop_config.sae_greedy_epochs,
        "sae_finetune_epochs": (training_loop_config.sae_finetune_epochs),
    }

    if data_validation_config is not None:
        params["validation_data"] = Path(data_validation_config.expression_file).name

    return params


def train(  # noqa: PLR0913
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    loss_config: LossConfig,
    training_loop_config: TrainingLoopConfig,
    data_training_config: DataConfig,
    data_validation_config: DataConfig | None = None,
    phase2_model_output: str | None = None,
) -> Autoencoder:
    """Train a SupCon autoencoder model.

    Args:
        model_config: Model configuration.
        optimizer_config: Optimizer configuration.
        loss_config: Loss configuration.
        training_loop_config: Training loop configuration.
        data_training_config: Training data configuration.
        data_validation_config: Validation data configuration.
        phase2_model_output: Optional path to save the Phase 2 model.

    Returns:
        Trained autoencoder model
    """
    training_loader = create_dataloader(data_training_config)
    validation_loader = None
    if data_validation_config is not None:
        validation_loader = create_dataloader(data_validation_config)

    # new tensor loaders (for Phases 1-2)
    tensor_train_loader = create_tensor_dataloader(data_training_config)
    tensor_val_loader = (
        create_tensor_dataloader(data_validation_config)
        if data_validation_config is not None
        else None
    )

    input_dim = training_loader.dataset[0]["features"].shape[0]

    logger.debug("Autoencoder input/output dimension: %d", input_dim)

    model = create_stacked_autoencoder(input_dim, model_config=model_config)
    model = model.to(torch.device(training_loop_config.device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config.learning_rate)

    # Create augmentation module
    augmentation_module = GeneExpressionAugmentation().to(
        torch.device(training_loop_config.device)
    )

    loss_fn = HybridLoss(
        sup_con_loss=SupConLoss(temperature=loss_config.supcon_temperature),
        reconstruction_loss=nn.MSELoss(),
        lambda_=loss_config.hybrid_lambda,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        augmentation_module=augmentation_module,
    )

    params = _build_params(
        model_config,
        optimizer_config,
        loss_config,
        training_loop_config,
        data_training_config,
        data_validation_config,
    )

    common_name = _generate_random_name()
    logger.info("Common experiment name: %s", common_name)

    # Phase 1
    with (
        StandardLoggingTracker(
            logger=logger,
            logging_interval=training_loop_config.sae_greedy_epochs // 10,
            experiment_steps=training_loop_config.sae_greedy_epochs,
        ) as logging_tracker_p1,
        MLflowTracker(
            experiment_name="gene-expression-augmented-denoise",
            run_name=f"phase1-{common_name}",
        ) as mlflow_tracker_p1,
    ):
        experiment_trackers_p1: list[ExperimentTracker] = [
            logging_tracker_p1,
            mlflow_tracker_p1,
        ]
        logging_tracker_p1.log_params(params)
        mlflow_tracker_p1.log_params(params)

        pretrain_phase1(
            model=model,
            optimizer=optimizer,
            train_loader=tensor_train_loader,
            val_loader=tensor_val_loader,
            sae_greedy_epochs=training_loop_config.sae_greedy_epochs,
            experiment_trackers=experiment_trackers_p1,
        )

    # Phase 2
    with (
        StandardLoggingTracker(
            logger=logger,
            logging_interval=training_loop_config.sae_finetune_epochs // 10,
            experiment_steps=training_loop_config.sae_finetune_epochs,
        ) as logging_tracker_p2,
        MLflowTracker(
            experiment_name="gene-expression-augmented-denoise",
            run_name=f"phase2-{common_name}",
        ) as mlflow_tracker_p2,
    ):
        experiment_trackers_p2: list[ExperimentTracker] = [
            logging_tracker_p2,
            mlflow_tracker_p2,
        ]
        logging_tracker_p2.log_params(params)
        mlflow_tracker_p2.log_params(params)

        pretrain_phase2(
            model=model,
            optimizer=optimizer,
            train_loader=tensor_train_loader,
            val_loader=tensor_val_loader,
            sae_finetune_epochs=training_loop_config.sae_finetune_epochs,
            experiment_trackers=experiment_trackers_p2,
        )

        if phase2_model_output is not None:
            model.save(phase2_model_output)
            logger.info("Phase 2 model saved to %s", phase2_model_output)

    # Phase 3
    with (
        StandardLoggingTracker(
            logger=logger,
            logging_interval=training_loop_config.supcon_hybrid_epochs // 10,
            experiment_steps=training_loop_config.supcon_hybrid_epochs,
        ) as logging_tracker_p3,
        MLflowTracker(
            experiment_name="gene-expression-augmented-denoise",
            run_name=f"phase3-{common_name}",
        ) as mlflow_tracker_p3,
    ):
        experiment_trackers_p3: list[ExperimentTracker] = [
            logging_tracker_p3,
            mlflow_tracker_p3,
        ]
        logging_tracker_p3.log_params(params)
        mlflow_tracker_p3.log_params(params)

        logger.info("Starting Phase 3: Hybrid loss fine-tuning")
        trainer.train(
            train_loader=training_loader,
            device=torch.device(training_loop_config.device),
            epochs=training_loop_config.supcon_hybrid_epochs,
            val_loader=validation_loader,
            experiment_trackers=experiment_trackers_p3,
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
        phase2_model_output=args.phase2_model_output,
    )

    if isinstance(model, (AutoEncoder, StackedAutoEncoder)):
        model.save(args.model_output)
    else:
        torch.save(model.state_dict(), args.model_output)
