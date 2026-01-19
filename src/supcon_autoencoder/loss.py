"""Custom loss function classes."""

from typing import Protocol

import torch
from torch import nn


class ReconstructionLossProtocol(Protocol):
    """Protocol for a reconstruction loss function."""

    def __call__(
        self, original_input: torch.Tensor, reconstructed_input: torch.Tensor
    ) -> torch.Tensor:
        """Compute the reconstruction loss.

        Args:
            original_input: Original input tensor.
            reconstructed_input: Reconstructed input tensor.

        Returns:
            torch.Tensor: The reconstruction loss value (scalar).
        """
        ...


class SupConLossProtocol(Protocol):
    """Protocol to lossly couple call signature of SupCon loss."""

    def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the SupCon loss.

        Args:
            embeddings: Embeddings (batch_size, latent_dim).
            labels: Labels (batch_size).

        Returns:
            torch.Tensor: The loss value (scalar).
        """
        ...


class HybridLoss(nn.Module):
    """Hybrid loss function combining SupCon loss and reconstruction loss as in SALSA.

    The loss is a weighted combination of the two losses:
    L = lambda * sup_con_loss + (1 - lambda) * reconstruction_loss
    """

    def __init__(
        self,
        sup_con_loss: SupConLossProtocol,
        reconstruction_loss: ReconstructionLossProtocol,
        lambda_: float = 0.5,
    ) -> None:
        """Initialize hybrid loss, combining SupCon loss and reconstruction loss.

        Args:
            sup_con_loss: SupCon (Supervised Contrastive) loss implementation.
            reconstruction_loss: Reconstruction loss implementation.
            lambda_: Weight for the SupCon loss and (1 - lambda) for reconstruction.
        """
        super().__init__()
        if not (0 <= lambda_ <= 1):
            msg = "lambda does not satisfy 0 <= lambda <= 1 condition"
            raise ValueError(msg)
        self.sup_con_loss = sup_con_loss
        self.reconstruction_loss = reconstruction_loss
        self.lambda_ = lambda_

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        original_input: torch.Tensor,
        reconstructed_input: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the hybrid loss.

        Args:
            embeddings: Embeddings for SupCon loss.
            labels: Labels for SupCon loss.
            original_input: Original input tensor.
            reconstructed_input: Reconstructed input tensor.

        Returns:
            torch.Tensor: The hybrid loss value (scalar).
        """
        sup_con = self.sup_con_loss(embeddings, labels)
        recon = self.reconstruction_loss(original_input, reconstructed_input)
        return self.lambda_ * sup_con + (1 - self.lambda_) * recon


class SupConLoss(nn.Module):
    """Outer Supervised Contrastive Loss as described in the SupCon paper.

    The loss encourages embeddings of the same class (positives) to be closer,
    while pushing apart embeddings of different classes (negatives).
    """

    def __init__(self, temperature: float = 0.7) -> None:
        """Initialize SupConLoss.

        Args:
            temperature: Temperature parameter τ.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the SupCon loss.

        Args:
            embeddings: Embeddings (batch_size, latent_dim).
            labels: Labels (batch_size).

        Returns:
            torch.Tensor: The loss value (scalar).
        """
        # Normalize embeddings to ensure cosine similarity
        embeddings = nn.functional.normalize(embeddings, dim=1)

        batch_size = embeddings.shape[0]

        # Compute similarity matrix: z_i · z_j / τ
        sim = embeddings @ embeddings.T / self.temperature  # [batch_size, batch_size]

        # Mask self-similarities
        mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positives mask: same label and not self
        labels = labels.unsqueeze(1)  # [batch_size, 1]
        pos_mask = (labels == labels.T) & ~mask  # [batch_size, batch_size]

        # Denominator: log sum_{a != i} exp(sim[i, a])
        den = torch.logsumexp(sim, dim=1)  # [batch_size]

        # Numerator sum: sum_{p in P(i)} sim[i, p]
        num_sims = (sim * pos_mask.float()).sum(dim=1)  # [batch_size]

        # Number of positives per anchor
        num_pos = pos_mask.sum(dim=1)  # [batch_size]

        # Mask out anchors with no positives (or only self)
        valid = num_pos > 0
        if not valid.any():
            return torch.tensor(
                1e-8, device=embeddings.device, requires_grad=embeddings.requires_grad
            )

        # Loss per anchor: - (1/|P(i)|) sum_p (sim[i,p] - den[i])
        loss_per_anchor = -(num_sims / num_pos) + den
        loss_per_anchor = loss_per_anchor[valid]

        # Total loss: average over batch
        return loss_per_anchor.mean()
