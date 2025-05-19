"""Centered Kernel Alignment (CKA) distortion metric."""

from torch import Tensor

from .distortion_metric import DistortionMetric
from .losses import Loss, NegativeInnerProduct


class CKA(DistortionMetric):
    """Centered Kernel Alignment (CKA) distortion metric.

    CKA measures the similarity between two sets of representations.
    See Similarity of Neural Network Representations Revisited (https://arxiv.org/abs/1905.00414)
    """

    loss: Loss = NegativeInnerProduct()

    def __init__(
        self,
        devide_by_std: bool = True,
    ):
        """Initialize CKA.

        Args:
            devide_by_std: Whether to divide the input by its standard deviation.
        """
        super().__init__()
        self.devide_by_std = devide_by_std

    def kernel(
        self,
        x: Tensor,
        other: Tensor | None = None,
    ) -> Tensor:
        """Compute the CKA kernel.

        Args:
            x: The first input tensor.
            other: The second input tensor. If None, x is used.

        Returns:
            The CKA kernel matrix.
        """
        if self.devide_by_std:
            x = x / x.std(dim=1, keepdim=True)

        if other is None:
            other = x
        elif self.devide_by_std:
            other = other / other.std(dim=1, keepdim=True)

        inner_product = x @ other.T
        centered = inner_product - inner_product.mean(dim=1, keepdim=True)
        scaled = centered / (centered * centered.T).sum().sqrt()
        return scaled

    kernel_source = kernel
    kernel_target = kernel

    def __str__(self):
        """Return the string representation of CKA."""
        return "CKA"
