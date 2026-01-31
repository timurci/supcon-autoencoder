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

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings to ensure cosine similarity.

        Args:
            embeddings: Raw embeddings (batch_size, latent_dim).

        Returns:
            torch.Tensor: Normalized embeddings (batch_size, latent_dim).
        """
        return nn.functional.normalize(embeddings, dim=1)

    def _compute_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute the similarity matrix with temperature scaling.

        Computes z_i · z_j / τ for all pairs.

        Args:
            embeddings: Normalized embeddings (batch_size, latent_dim).

        Returns:
            torch.Tensor: Similarity matrix (batch_size, batch_size).
        """
        return embeddings @ embeddings.T / self.temperature

    def _create_masks(
        self, batch_size: int, labels: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create self-similarity mask and positives mask.

        Args:
            batch_size: Number of samples in batch.
            labels: Labels for each sample (batch_size).
            device: Device to create masks on.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Self-similarity mask and positives mask,
                both of shape (batch_size, batch_size).
        """
        # Self-similarity mask
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # Positives mask: same label and not self
        labels = labels.unsqueeze(1)  # [batch_size, 1]
        pos_mask = (labels == labels.T) & ~self_mask  # [batch_size, batch_size]

        return self_mask, pos_mask

    def _compute_denominator(self, sim: torch.Tensor) -> torch.Tensor:
        """Compute the denominator: log sum_{a != i} exp(sim[i, a]).

        Args:
            sim: Similarity matrix with self-similarities masked as -inf.

        Returns:
            torch.Tensor: Denominator per anchor (batch_size).
        """
        return torch.logsumexp(sim, dim=1)

    def _compute_numerator(
        self, sim: torch.Tensor, pos_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute the numerator: sum_{p in P(i)} sim[i, p].

        Args:
            sim: Similarity matrix (batch_size, batch_size).
            pos_mask: Positives mask (batch_size, batch_size).

        Returns:
            torch.Tensor: Sum of positive similarities per anchor (batch_size).
        """
        return sim.masked_fill(~pos_mask, 0.0).sum(dim=1)

    def _compute_loss_per_anchor(
        self,
        num_sims: torch.Tensor,
        num_pos: torch.Tensor,
        den: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute loss per anchor.

        Loss per anchor: - (1/|P(i)|) sum_p (sim[i,p] - den[i])

        Args:
            num_sims: Sum of positive similarities per anchor (batch_size).
            num_pos: Number of positives per anchor (batch_size).
            den: Denominator per anchor (batch_size).

        Returns:
            torch.Tensor | None: Loss per anchor for valid anchors,
                or None if no valid anchors exist.
        """
        # Mask out anchors with no positives (or only self)
        valid = num_pos > 0
        if not valid.any():
            return None

        # Loss per anchor: - (1/|P(i)|) sum_p (sim[i,p] - den[i])
        loss_per_anchor = -(num_sims / num_pos) + den
        return loss_per_anchor[valid]

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the SupCon loss.

        Args:
            embeddings: Embeddings (batch_size, latent_dim).
            labels: Labels (batch_size).

        Returns:
            torch.Tensor: The loss value (scalar).
        """
        # Step 1: Normalize embeddings
        embeddings = self._normalize_embeddings(embeddings)

        batch_size = embeddings.shape[0]

        # Step 2: Compute similarity matrix
        sim = self._compute_similarity_matrix(embeddings)

        # Step 3: Create masks
        self_mask, pos_mask = self._create_masks(batch_size, labels, embeddings.device)

        # Step 4: Mask self-similarities
        sim = sim.masked_fill(self_mask, float("-inf"))

        # Step 5: Compute denominator
        den = self._compute_denominator(sim)

        # Step 6: Compute numerator
        num_sims = self._compute_numerator(sim, pos_mask)

        # Step 7: Count positives per anchor
        num_pos = pos_mask.sum(dim=1)

        # Step 8: Compute loss per anchor
        loss_per_anchor = self._compute_loss_per_anchor(num_sims, num_pos, den)

        if loss_per_anchor is None:
            return torch.tensor(
                1e-8, device=embeddings.device, requires_grad=embeddings.requires_grad
            )

        # Step 9: Total loss: average over batch
        return loss_per_anchor.mean()
