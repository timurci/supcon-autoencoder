"""Unit tests for loss functions."""

import pytest
import torch
from torch import nn

from supcon_autoencoder.core.loss import HybridLoss, SupConLoss


class TestHybridLoss:
    """Test suite for HybridLoss class."""

    def test_lambda_validation_error(self) -> None:
        """Test lambda validation raises ValueError for invalid values."""
        mock_sup = nn.MSELoss()
        mock_recon = nn.MSELoss()

        # Test negative lambda
        with pytest.raises(ValueError, match="lambda does not satisfy"):
            HybridLoss(mock_sup, mock_recon, lambda_=-0.1)

        # Test lambda > 1
        with pytest.raises(ValueError, match="lambda does not satisfy"):
            HybridLoss(mock_sup, mock_recon, lambda_=1.5)

    def test_lambda_weighting(self) -> None:
        """Test that lambda properly weights the two losses with varying values."""
        # Use real loss functions with specific inputs for exact numerical validation
        sup_loss = SupConLoss(temperature=0.5)
        recon_loss = nn.MSELoss()

        # Complex 3D embeddings with varying magnitudes (2 samples, same class)
        embeddings = torch.tensor(
            [
                [1.0, 0.5, -0.2],
                [0.9, 0.6, -0.1],
            ]
        )
        labels = torch.tensor([0, 0])

        # Varying reconstruction values
        original = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )
        reconstructed = torch.tensor(
            [
                [1.5, 2.5, 2.5, 4.5],
                [4.5, 6.5, 6.5, 7.5],
            ]
        )

        # Test with lambda = 0.3 (unequal weighting)
        hybrid = HybridLoss(sup_loss, recon_loss, lambda_=0.3)
        loss = hybrid(embeddings, labels, original, reconstructed)

        expected = 0.175
        assert abs(loss.item() - expected) < 1e-6

        # Test with lambda = 0.0 (only reconstruction)
        hybrid0 = HybridLoss(sup_loss, recon_loss, lambda_=0.0)
        loss0 = hybrid0(embeddings, labels, original, reconstructed)
        assert abs(loss0.item() - 0.25) < 1e-6

        # Test with lambda = 1.0 (only supcon)
        hybrid1 = HybridLoss(sup_loss, recon_loss, lambda_=1.0)
        loss1 = hybrid1(embeddings, labels, original, reconstructed)
        assert abs(loss1.item()) < 1e-6

    def test_hybrid_loss_numerical_exact(self) -> None:
        """Test hybrid loss with hardcoded values for exact numerical validation."""
        # Use simple embeddings and reconstruction for exact computation
        # Embeddings: orthogonal unit vectors in 2D, two classes
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],  # Same as first (class 0)
                [0.0, 1.0],
                [0.0, 1.0],  # Same as third (class 1)
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])

        # Original and reconstructed for MSE (simple difference)
        original = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        )
        reconstructed = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        )

        sup_loss = SupConLoss(temperature=1.0)
        recon_loss = nn.MSELoss()
        hybrid = HybridLoss(sup_loss, recon_loss, lambda_=0.5)

        loss = hybrid(embeddings, labels, original, reconstructed)

        # Manually computed: 0.5 * 0.5514447093 (SupCon) + 0.5 * 0.0 (MSE)
        expected_loss = 0.27572235465

        assert abs(loss.item() - expected_loss) < 1e-6

    def test_hybrid_loss_numerical_complex(self) -> None:
        """Test hybrid loss with complex 6-sample scenario."""
        # Complex 3D embeddings with 3 classes, varying magnitudes
        embeddings = torch.tensor(
            [
                [2.0, 1.0, 0.5],
                [1.8, 1.2, 0.3],
                [-1.0, 2.0, 0.0],
                [-0.8, 1.9, 0.2],
                [0.0, -1.5, 2.0],
                [0.2, -1.3, 1.8],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1, 2, 2])

        # Original and reconstructed with small differences (non-zero MSE)
        original = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
                [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
            ]
        )
        reconstructed = torch.tensor(
            [
                [1.2, 2.1, 2.8, 4.2, 4.9, 6.1],
                [7.2, 7.9, 9.2, 9.8, 11.2, 11.9],
                [13.1, 14.2, 14.9, 16.1, 17.2, 17.9],
                [19.2, 19.9, 21.1, 22.2, 22.9, 24.1],
                [25.1, 26.2, 26.9, 28.1, 29.2, 29.9],
                [31.2, 31.9, 33.1, 34.2, 34.9, 36.1],
            ]
        )

        sup_loss = SupConLoss(temperature=1.0)
        recon_loss = nn.MSELoss()

        # Test with lambda = 0.7 (unequal weighting)
        hybrid = HybridLoss(sup_loss, recon_loss, lambda_=0.7)
        loss = hybrid(embeddings, labels, original, reconstructed)

        # Manually computed: 0.7 * 0.8250042796 + 0.3 * 0.0225000475
        expected_loss = 0.5842530100

        assert abs(loss.item() - expected_loss) < 1e-6


class TestSupConLoss:
    """Test suite for SupConLoss class."""

    def test_numerical_exact_simple_case(self) -> None:
        """Test SupCon loss with hardcoded embeddings for exact numerical validation."""
        # Simple case: orthogonal unit vectors, two samples per class
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])

        # Test with temperature = 1.0
        loss_fn = SupConLoss(temperature=1.0)
        loss = loss_fn(embeddings, labels)

        # Manually computed expected value
        expected = 0.5514447093
        assert abs(loss.item() - expected) < 1e-6

        # Test with temperature = 0.5 (similarities doubled)
        loss_fn_half = SupConLoss(temperature=0.5)
        loss_half = loss_fn_half(embeddings, labels)

        expected_half = 0.2395447662
        assert abs(loss_half.item() - expected_half) < 1e-6

    def test_numerical_exact_complex_3d(self) -> None:
        """Test SupCon loss with complex 3D embeddings and varying similarities."""
        # Complex 3D embeddings with 3 classes, varying magnitudes and directions
        embeddings = torch.tensor(
            [
                [3.0, 1.0, 0.0],
                [2.0, 1.5, 0.5],
                [-2.0, 1.0, 0.0],
                [-1.5, 1.2, 0.3],
                [0.0, -2.0, 1.0],
                [0.2, -1.8, 0.8],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1, 2, 2])

        loss_fn = SupConLoss(temperature=1.0)
        loss = loss_fn(embeddings, labels)

        # Manually computed expected value
        expected = 0.6954500675
        assert abs(loss.item() - expected) < 1e-6

    def test_forward_no_positives(self) -> None:
        """Test forward pass when samples have no positives (all different classes)."""
        loss_fn = SupConLoss(temperature=0.5)

        embeddings = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 3])  # All different

        loss = loss_fn(embeddings, labels)

        # Should return small positive value when no positives exist
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
        assert loss.item() < 1e-7

    def test_forward_single_sample(self) -> None:
        """Test forward pass with single sample."""
        loss_fn = SupConLoss(temperature=0.5)

        embeddings = torch.randn(1, 8)
        labels = torch.tensor([0])

        loss = loss_fn(embeddings, labels)

        # Should return small positive value
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
        assert loss.item() < 1e-7

    def test_forward_all_same_class(self) -> None:
        """Test forward pass when all samples have same class."""
        loss_fn = SupConLoss(temperature=0.5)

        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
            ],
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 0, 0])

        loss = loss_fn(embeddings, labels)

        # Should compute loss successfully
        assert loss.shape == torch.Size([])
        assert loss.requires_grad

    def test_forward_gradient_flow(self) -> None:
        """Test that gradients flow through the loss."""
        loss_fn = SupConLoss(temperature=0.5)

        embeddings = torch.randn(4, 8, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])

        loss = loss_fn(embeddings, labels)
        loss.backward()

        # Gradients should exist and be non-zero
        assert embeddings.grad is not None
        assert not torch.all(embeddings.grad == 0)

    def test_temperature_effect(self) -> None:
        """Test that temperature affects the loss computation with varying values."""
        # Complex 3D embeddings with varying magnitudes across 3 classes
        embeddings = torch.tensor(
            [
                [2.0, 1.0, 0.5],
                [1.8, 1.2, 0.3],
                [-1.0, 2.0, 0.0],
                [-0.8, 1.9, 0.2],
                [0.0, -1.5, 2.0],
                [0.2, -1.3, 1.8],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1, 2, 2])

        # Test with three different temperatures
        loss_fn_05 = SupConLoss(temperature=0.5)
        loss_fn_10 = SupConLoss(temperature=1.0)
        loss_fn_20 = SupConLoss(temperature=2.0)

        loss_05 = loss_fn_05(embeddings, labels)
        loss_10 = loss_fn_10(embeddings, labels)
        loss_20 = loss_fn_20(embeddings, labels)

        # All three should be different
        assert abs(loss_05.item() - loss_10.item()) > 0.1
        assert abs(loss_10.item() - loss_20.item()) > 0.1
        # Lower temperature should give lower loss (sharper contrasts)
        assert loss_05.item() < loss_10.item() < loss_20.item()

    def test_device_compatibility(self) -> None:
        """Test that loss works on different devices if available."""
        loss_fn = SupConLoss(temperature=0.5)

        embeddings = torch.randn(4, 8)
        labels = torch.tensor([0, 0, 1, 1])

        loss = loss_fn(embeddings, labels)

        # Should work on CPU
        assert loss.device == embeddings.device


class TestSupConLossModularMethods:
    """Test suite for SupConLoss modular helper methods."""

    def test_normalize_embeddings(self) -> None:
        """Test _normalize_embeddings produces unit vectors."""
        loss_fn = SupConLoss()

        embeddings = torch.tensor(
            [
                [3.0, 4.0],
                [1.0, 0.0],
                [0.0, 2.0],
            ]
        )

        normalized = loss_fn._normalize_embeddings(embeddings)

        # Check that each row has unit norm
        norms = torch.norm(normalized, dim=1)
        expected = torch.ones(3)
        assert torch.allclose(norms, expected, atol=1e-6)

    def test_compute_similarity_matrix(self) -> None:
        """Test _compute_similarity_matrix computation."""
        loss_fn = SupConLoss(temperature=0.5)

        # Orthogonal unit vectors
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        sim_matrix = loss_fn._compute_similarity_matrix(embeddings)

        # Diagonal should be 1/τ (same vector)
        expected_diag = 1.0 / 0.5
        assert abs(sim_matrix[0, 0].item() - expected_diag) < 1e-5
        assert abs(sim_matrix[1, 1].item() - expected_diag) < 1e-5

        # Off-diagonal should be 0 (orthogonal vectors)
        assert abs(sim_matrix[0, 1].item()) < 1e-5
        assert abs(sim_matrix[1, 0].item()) < 1e-5

    def test_create_masks(self) -> None:
        """Test _create_masks creates correct masks."""
        loss_fn = SupConLoss()

        batch_size = 4
        labels = torch.tensor([0, 0, 1, 1])
        device = labels.device

        self_mask, pos_mask = loss_fn._create_masks(batch_size, labels, device)

        # Self mask should be identity
        expected_self = torch.eye(4, dtype=torch.bool)
        assert torch.equal(self_mask, expected_self)

        # Pos mask should have same class but not self
        expected_pos = torch.tensor(
            [
                [False, True, False, False],
                [True, False, False, False],
                [False, False, False, True],
                [False, False, True, False],
            ]
        )
        assert torch.equal(pos_mask, expected_pos)

    def test_compute_denominator(self) -> None:
        """Test _compute_denominator with known values."""
        loss_fn = SupConLoss()

        # Simple similarity matrix with self similarities masked
        sim = torch.tensor(
            [
                [float("-inf"), 1.0, 2.0],
                [1.0, float("-inf"), 3.0],
                [2.0, 3.0, float("-inf")],
            ]
        )

        den = loss_fn._compute_denominator(sim)

        # Expected: log(exp(1) + exp(2)) for first row
        expected_0 = torch.log(
            torch.exp(torch.tensor(1.0)) + torch.exp(torch.tensor(2.0))
        )
        assert abs(den[0].item() - expected_0.item()) < 1e-5

    def test_compute_numerator(self) -> None:
        """Test _compute_numerator computation."""
        loss_fn = SupConLoss()

        sim = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        pos_mask = torch.tensor(
            [
                [False, True, False],
                [True, False, True],
                [False, False, False],
            ]
        )

        num = loss_fn._compute_numerator(sim, pos_mask)

        # First row: only second element is positive -> 2.0
        assert abs(num[0].item() - 2.0) < 1e-6
        # Second row: first and third are positive -> 4.0 + 6.0 = 10.0
        assert abs(num[1].item() - 10.0) < 1e-6
        # Third row: no positives -> 0.0
        assert abs(num[2].item()) < 1e-6

    def test_integration_modular_methods(self) -> None:
        """Test that modular methods integrate correctly in forward pass."""
        loss_fn = SupConLoss(temperature=1.0)

        # Simple test case
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],  # Same as first
            ]
        )
        labels = torch.tensor([0, 0])

        # Run through modular steps manually
        norm_emb = loss_fn._normalize_embeddings(embeddings)
        sim = loss_fn._compute_similarity_matrix(norm_emb)
        self_mask, pos_mask = loss_fn._create_masks(2, labels, embeddings.device)
        sim_masked = sim.masked_fill(self_mask, float("-inf"))
        den = loss_fn._compute_denominator(sim_masked)
        num_sims = loss_fn._compute_numerator(sim, pos_mask)
        num_pos = pos_mask.sum(dim=1)
        loss_per_anchor = loss_fn._compute_loss_per_anchor(num_sims, num_pos, den)

        assert loss_per_anchor is not None
        assert len(loss_per_anchor) == 2

        # Compare with full forward pass
        full_loss = loss_fn(embeddings, labels)
        manual_loss = loss_per_anchor.mean()

        assert torch.allclose(full_loss, manual_loss, atol=1e-6)
