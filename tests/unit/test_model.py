"""Unit tests for model module."""

import torch
from torch import nn

from supcon_autoencoder.core.model import (
    AugmentationResult,
    augment_samples_with_labels,
)


class MockAugmentationModule(nn.Module):
    """Mock augmentation module for testing.

    Simulates an augmentation that creates n_augmentations augmented versions
    per input sample. Includes mathematical operations that would create
    gradients outside of torch.no_grad().
    """

    def __init__(self, n_augmentations: int = 2) -> None:
        """Initialize mock augmentation module.

        Args:
            n_augmentations: Number of augmentations per sample.
        """
        super().__init__()
        self.n_augmentations = n_augmentations

    def forward(self, x: torch.Tensor) -> AugmentationResult:
        """Apply mock augmentation with mathematical operations.

        Performs operations that would create gradients if not in
        torch.no_grad() context.

        Args:
            x: Input tensor (batch_size, ...).

        Returns:
            AugmentationResult with augmented outputs and sample indices.
        """
        batch_size = x.shape[0]

        # Create n_augmentations copies of each sample
        augmented = x.repeat_interleave(self.n_augmentations, dim=0)

        # Each augmented sample maps back to its original index
        sample_indices = torch.arange(batch_size).repeat_interleave(
            self.n_augmentations
        )

        return {
            "outputs": augmented,
            "sample_indices": sample_indices,
        }


class TestAugmentSamplesWithLabels:
    """Test suite for augment_samples_with_labels function."""

    def test_three_augmentations(self) -> None:
        """Test with 3 augmentations per sample."""
        inputs = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        labels = torch.tensor([6, 2, 8])

        mock_aug = MockAugmentationModule(n_augmentations=3)

        augmented_inputs, augmented_labels = augment_samples_with_labels(
            mock_aug, inputs, labels
        )

        # Should have 3 * 3 = 9 samples
        assert augmented_inputs.shape == torch.Size([9, 2])
        assert augmented_labels.shape == torch.Size([9])

        # Check labels: each original label repeated 3 times
        expected_labels = torch.tensor([6, 6, 6, 2, 2, 2, 8, 8, 8])
        assert torch.equal(augmented_labels, expected_labels)

    def test_single_sample(self) -> None:
        """Test with single sample."""
        inputs = torch.tensor([[1.0, 2.0, 3.0]])
        labels = torch.tensor([5])

        mock_aug = MockAugmentationModule(n_augmentations=3)

        augmented_inputs, augmented_labels = augment_samples_with_labels(
            mock_aug, inputs, labels
        )

        # Should have 1 * 3 = 3 samples
        assert augmented_inputs.shape == torch.Size([3, 3])
        assert augmented_labels.shape == torch.Size([3])

        # All labels should be 5
        expected_labels = torch.tensor([5, 5, 5])
        assert torch.equal(augmented_labels, expected_labels)

    def test_empty_batch(self) -> None:
        """Test with empty batch."""
        inputs = torch.tensor([]).reshape(0, 2)
        labels = torch.tensor([])

        mock_aug = MockAugmentationModule(n_augmentations=2)

        augmented_inputs, _ = augment_samples_with_labels(mock_aug, inputs, labels)

        assert augmented_inputs.shape == torch.Size([0, 2])

    def test_different_input_shapes(self) -> None:
        """Test with different input tensor shapes."""
        mock_aug = MockAugmentationModule(n_augmentations=2)

        # 1D inputs
        inputs_1d = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        labels_1d = torch.tensor([1, 2, 3, 4])

        augmented_inputs_1d, augmented_labels_1d = augment_samples_with_labels(
            mock_aug, inputs_1d, labels_1d
        )

        assert augmented_inputs_1d.shape == torch.Size([8, 1])
        expected_labels = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
        assert torch.equal(augmented_labels_1d, expected_labels)

        # 3D inputs (e.g., sequences)
        inputs_3d = torch.randn(2, 4, 5)
        labels_3d = torch.tensor([10, 20])

        augmented_inputs_3d, augmented_labels_3d = augment_samples_with_labels(
            mock_aug, inputs_3d, labels_3d
        )

        assert augmented_inputs_3d.shape == torch.Size([4, 4, 5])
        expected_labels_3d = torch.tensor([10, 10, 20, 20])
        assert torch.equal(augmented_labels_3d, expected_labels_3d)

    def test_no_gradient_computation(self) -> None:
        """Test that augmentation doesn't compute gradients."""
        inputs = torch.tensor([[1.0, 2.0]], requires_grad=True)
        labels = torch.tensor([42])

        mock_aug = MockAugmentationModule(n_augmentations=2)

        augmented_inputs, _ = augment_samples_with_labels(mock_aug, inputs, labels)

        # Outputs should not require gradients (inside torch.no_grad())
        assert not augmented_inputs.requires_grad
