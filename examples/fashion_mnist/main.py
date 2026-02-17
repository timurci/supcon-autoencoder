"""Training script for SupCon autoencoder on Fashion-MNIST.

This is a simple, hardcoded example demonstrating how to train a supervised
contrastive autoencoder on the Fashion-MNIST dataset.
"""

import logging
from argparse import ArgumentParser

import torch
from dec_torch.autoencoder import AutoEncoder, AutoEncoderConfig
from torch import nn

from supcon_autoencoder.core.loss import HybridLoss, SupConLoss
from supcon_autoencoder.core.trackers import MLflowTracker, StandardLoggingTracker
from supcon_autoencoder.core.training import Trainer

from .dataset import create_dataloader

# Data parameters
BATCH_SIZE = 256

# Model parameters
INPUT_DIM = 784  # 28x28 flattened
LATENT_DIM = 32
HIDDEN_DIMS = [512, 256, 128]

# Training parameters
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGING_INTERVAL = EPOCHS // 10

# Loss parameters
SUPCON_TEMPERATURE = 0.5
HYBRID_LAMBDA = 0.5


def build_parser() -> ArgumentParser:
    """Create argument parser."""
    parser = ArgumentParser()
    parser.add_argument(
        "--model-output", required=True, help="Path to save the trained model."
    )
    return parser


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train() -> nn.Module:
    """Train a SupCon autoencoder on Fashion-MNIST.

    Returns:
        Trained autoencoder model
    """
    logger.info("Using device: %s", DEVICE)

    # Create dataloaders
    logger.info("Loading Fashion-MNIST dataset...")
    train_loader = create_dataloader(
        root="./data",
        batch_size=BATCH_SIZE,
        train=True,
        download=True,
        shuffle=True,
    )
    val_loader = create_dataloader(
        root="./data",
        batch_size=BATCH_SIZE,
        train=False,
        download=True,
        shuffle=False,
    )
    logger.info(
        "Dataset loaded - Train: %d samples, Val: %d samples",
        len(train_loader.dataset),  # type: ignore[arg-type]
        len(val_loader.dataset),  # type: ignore[arg-type]
    )

    # Create model
    logger.info(
        "Creating autoencoder with input_dim=%d, latent_dim=%d, hidden_dims=%s",
        INPUT_DIM,
        LATENT_DIM,
        HIDDEN_DIMS,
    )
    config = AutoEncoderConfig.build(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
    )
    model = AutoEncoder(config)
    model = model.to(DEVICE)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Create loss function
    loss_fn = HybridLoss(
        sup_con_loss=SupConLoss(temperature=SUPCON_TEMPERATURE),
        reconstruction_loss=nn.MSELoss(),
        lambda_=HYBRID_LAMBDA,
    )

    # Create trainer
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)

    # Define parameters for tracking
    params = {
        "dataset": "Fashion-MNIST",
        "batch_size": BATCH_SIZE,
        "input_dim": INPUT_DIM,
        "latent_dim": LATENT_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": str(optimizer),
        "supcon_temperature": SUPCON_TEMPERATURE,
        "hybrid_lambda": HYBRID_LAMBDA,
    }

    logger.info("Starting training for %d epochs...", EPOCHS)

    with (
        StandardLoggingTracker(
            logger=logger, logging_interval=LOGGING_INTERVAL, experiment_steps=EPOCHS
        ) as logging_tracker,
        MLflowTracker(
            experiment_name="fashion-mnist-supcon-autoencoder"
        ) as mlflow_tracker,
    ):
        logging_tracker.log_params(params)
        mlflow_tracker.log_params(params)

        trainer.train(
            train_loader=train_loader,
            device=DEVICE,
            epochs=EPOCHS,
            val_loader=val_loader,
            experiment_trackers=[logging_tracker, mlflow_tracker],
        )

    return model


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Train model
    model = train()

    # Save model
    if isinstance(model, AutoEncoder):
        model.save(args.model_output)
    else:
        torch.save(model.state_dict(), args.model_output)
    logger.info("Model saved to %s", args.model_output)

    logger.info("Training complete!")
