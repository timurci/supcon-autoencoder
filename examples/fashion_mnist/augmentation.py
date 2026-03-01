"""Image augmentation module for Fashion-MNIST dataset."""

import torch
from torch import nn
from torch.nn import functional as f


class FashionMNISTAugmentation(nn.Module):
    """Augmentation module for Fashion-MNIST images.

    Creates 3 views per image:
    - Original (unmodified)
    - Augmentation 1: rotation + flip + noise
    - Augmentation 2: translation + flip + noise

    Output is batch_size * 3 samples.
    """

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply augmentations to input images.

        Args:
            x: Input tensor of shape (batch_size, 784) - flattened 28x28 images.

        Returns:
            Dict with keys:
                - 'outputs': Augmented images (batch_size * 3, 784)
                - 'sample_indices': Tensor mapping each output to
                  original index (batch_size * 3,)
        """
        batch_size = x.size(0)
        device = x.device

        # Reshape to (batch_size, 1, 28, 28) for image operations
        images = x.view(batch_size, 1, 28, 28)

        # Create sample indices: each original index repeated 3 times
        sample_indices = torch.arange(batch_size, device=device).repeat_interleave(3)

        # Augmentation 1: rotation + flip + noise
        aug1 = self._augment_rotation_flip_noise(images, device)

        # Augmentation 2: translation + flip + noise
        aug2 = self._augment_translation_flip_noise(images, device)

        # Flatten all to (batch_size, 784)
        original_flat = x  # already flat
        aug1_flat = aug1.view(batch_size, 784)
        aug2_flat = aug2.view(batch_size, 784)

        # Interleave original and augmentations: orig0, aug1_0, aug2_0, orig1, ...
        outputs = torch.empty(batch_size * 3, 784, device=device)
        outputs[0::3] = original_flat
        outputs[1::3] = aug1_flat
        outputs[2::3] = aug2_flat

        return {"outputs": outputs, "sample_indices": sample_indices}

    def _augment_rotation_flip_noise(
        self, images: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Apply rotation, random flip, and noise."""
        aug = images.clone()

        # Random horizontal flip (50% probability)
        if torch.rand(1).item() > 0.5:  # noqa: PLR2004
            aug = torch.flip(aug, dims=[3])

        # Random rotation (-10 to +10 degrees)
        angle = torch.rand(1).item() * 20 - 10
        angle_rad = torch.tensor(angle * 3.14159 / 180)
        theta = (
            torch.tensor(
                [
                    [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
                    [torch.sin(angle_rad), torch.cos(angle_rad), 0],
                ],
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
            .repeat(aug.size(0), 1, 1)
        )

        grid = f.affine_grid(theta, list(aug.size()), align_corners=False)
        aug = f.grid_sample(
            aug, grid, align_corners=False, mode="bilinear", padding_mode="zeros"
        )

        # Gaussian noise
        noise_std = torch.rand(1).item() * 0.05
        noise = torch.randn_like(aug) * noise_std
        aug = aug + noise
        return torch.clamp(aug, 0, 1)

    def _augment_translation_flip_noise(
        self, images: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Apply translation, random flip, and noise."""
        aug = images.clone()

        # Random horizontal flip (50% probability)
        if torch.rand(1).item() > 0.5:  # noqa: PLR2004
            aug = torch.flip(aug, dims=[3])

        # Random translation (-10% to +10%)
        tx = torch.rand(1).item() * 0.2 - 0.1
        ty = torch.rand(1).item() * 0.2 - 0.1
        theta_translate = (
            torch.tensor([[1, 0, tx], [0, 1, ty]], dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(aug.size(0), 1, 1)
        )

        grid_translate = f.affine_grid(
            theta_translate, list(aug.size()), align_corners=False
        )
        aug = f.grid_sample(
            aug,
            grid_translate,
            align_corners=False,
            mode="bilinear",
            padding_mode="zeros",
        )

        # Gaussian noise
        noise_std = torch.rand(1).item() * 0.05
        noise = torch.randn_like(aug) * noise_std
        aug = aug + noise
        return torch.clamp(aug, 0, 1)
