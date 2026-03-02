"""Gene expression augmentation module."""

import torch
from torch import nn


class GeneExpressionAugmentation(nn.Module):
    """Augmentation module for gene expression data.

    Creates 3 views per sample:
    - Original (unmodified)
    - Augmentation 1: Gaussian noise
    - Augmentation 2: Poisson noise

    Output is batch_size * 3 samples.
    """

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply augmentations to input gene expression data.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Dict with keys:
                - 'outputs': Augmented samples (batch_size * 3, input_dim)
                - 'sample_indices': Tensor mapping each output to
                  original index (batch_size * 3,)
        """
        batch_size = x.size(0)
        device = x.device

        # Augmentation 1: Gaussian noise
        aug1 = self._augment_gaussian_noise(x)

        # Augmentation 2: Poisson noise
        aug2 = self._augment_poisson_noise(x)

        # Concatenate original and augmentations
        outputs = torch.cat([x, aug1, aug2], dim=0)
        sample_indices = torch.arange(batch_size, device=device).repeat(3)

        return {"outputs": outputs, "sample_indices": sample_indices}

    def _augment_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise augmentation.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Augmented tensor with Gaussian noise added.
        """
        batch_size = x.size(0)
        device = x.device
        # Random noise standard deviation (0.05 to 0.2) per sample and feature
        noise_std = 0.05 + torch.rand(batch_size, x.size(1), device=device) * 0.15
        noise = torch.randn_like(x) * noise_std
        return x + noise

    def _augment_poisson_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Poisson noise augmentation.

        For gene expression data, we simulate Poisson noise by treating
        the values as rates/lambda parameters. We use torch.poisson which
        generates Poisson-distributed samples.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Augmented tensor with Poisson noise.
        """
        x_positive = torch.clamp(x, min=0.0)
        batch_size = x.size(0)
        device = x.device
        scale = 0.5 + torch.rand(batch_size, x.size(1), device=device) * 0.5
        lambda_param = x_positive * scale + 1e-8
        poisson_samples = torch.poisson(lambda_param)
        return poisson_samples / scale
