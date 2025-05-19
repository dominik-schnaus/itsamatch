"""Gromov-Wasserstein distortion metric."""

from torch import Tensor, cdist, half

from .distortion_metric import DistortionMetric
from .losses import Loss, SquaredLoss


class GromovWasserstein(DistortionMetric):
    """Gromov-Wasserstein (GW) distortion metric.

    The GW distance measures the dissimilarity between two metric measure spaces.
    In this context, the kernel is the pairwise Euclidean distance matrix and the
    loss is the squared loss.

    See "Gromov–Wasserstein distances and the metric approach to object matching"
    (https://media.adelaide.edu.au/acvt/Publications/2011/2011-Gromov%E2%80%93Wasserstein%20Distances%20and%20the%20Metric%20Approach%20to%20Object%20Matching.pdf).
    """

    loss: Loss = SquaredLoss()

    def kernel(
        self,
        x: Tensor,
        other: Tensor | None = None,
    ) -> Tensor:
        """Compute the pairwise Euclidean distance matrix.

        Args:
            x: The first input tensor of shape (n_samples_x, n_features).
            other: The second input tensor of shape (n_samples_y, n_features).
                   If None, x is used.

        Returns:
            The pairwise Euclidean distance matrix of shape (n_samples_x, n_samples_y).
        """
        if other is None:
            other = x
        dtype = x.dtype
        if x.dtype == half:
            x = x.float()
        if other.dtype == half:
            other = other.float()

        return cdist(x, other, p=2).to(dtype=dtype)

    kernel_source = kernel
    kernel_target = kernel

    def __str__(self):
        """Return the string representation of the GromovWasserstein metric."""
        return "GW Distance"
