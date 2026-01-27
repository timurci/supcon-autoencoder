"""Module to create autoencoder models."""

from typing import TYPE_CHECKING

from dec_torch.autoencoder import AutoEncoder, AutoEncoderConfig

if TYPE_CHECKING:
    from torch import nn

    from examples.gene_expression.config import ModelConfig


def create_autoencoder(input_dim: int, model_config: ModelConfig) -> nn.Module:
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
