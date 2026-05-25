"""Module to create autoencoder models."""

from typing import TYPE_CHECKING

from dec_torch.autoencoder import (
    AutoEncoder,
    AutoEncoderConfig,
    StackedAutoEncoder,
    StackedAutoEncoderConfig,
)

if TYPE_CHECKING:
    from supcon_autoencoder.core.model import Autoencoder

    from .config import ModelConfig


def create_autoencoder(input_dim: int, model_config: ModelConfig) -> Autoencoder:
    """Create an autoencoder model from model config."""
    config = AutoEncoderConfig.build(
        input_dim=input_dim,
        latent_dim=model_config.latent_dim,
        hidden_dims=model_config.hidden_dims,
        input_dropout=model_config.input_dropout,
        hidden_activation=model_config.hidden_activation,
        encoder_output_activation=model_config.encoder_activation,
        decoder_output_activation=model_config.decoder_activation,
    )
    return AutoEncoder(config)


def create_stacked_autoencoder(
    input_dim: int, model_config: ModelConfig
) -> StackedAutoEncoder:
    """Create a stacked autoencoder from model config.

    ModelConfig.hidden_dims defines the intermediate latent dimensions of the stack.
    ModelConfig.latent_dim is the final bottleneck.
    """
    latent_dims = (model_config.hidden_dims or []) + [model_config.latent_dim]
    if not latent_dims:
        msg = "At least one latent dimension is required for stacked autoencoder"
        raise ValueError(msg)

    config = StackedAutoEncoderConfig.build(
        input_dim=input_dim,
        latent_dims=latent_dims,
        hidden_dims=None,  # no hidden layers within each sub-autoencoder
        input_dropout=model_config.input_dropout,
        hidden_activation=model_config.hidden_activation,
        last_encoder_activation=model_config.encoder_activation,
        last_decoder_activation=model_config.decoder_activation,
    )
    return StackedAutoEncoder(config)
