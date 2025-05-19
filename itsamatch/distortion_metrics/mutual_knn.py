"""Mutual k-Nearest Neighbors (k-NN) distortion metric."""

from math import sqrt

import torch
from torch import Tensor

from .distortion_metric import DistortionMetric
from .losses import Loss, NegativeInnerProduct


def jaccard(x: Tensor, other: Tensor | None = None) -> Tensor:
    """
    Calculate the Jaccard index (Intersection over Union - IoU) between sets of labels.

    If `other` is None, computes the Jaccard index between all pairs of rows in `x`.
    Otherwise, computes the Jaccard index between rows of `x` and rows of `other`.
    Assumes that labels are integers and -1 is a special value to be ignored.

    Args:
        x: A 2D tensor where each row represents a set of labels.
        other: An optional 2D tensor with the same structure as `x`.

    Returns:
        A 2D tensor containing the Jaccard index scores.
    """
    if other is None:
        values, inverse = x.unique(return_inverse=True)
        one_hot_combined = torch.zeros(
            x.shape[0], values.shape[0], dtype=torch.bool, device=x.device
        )
        for el in inverse.permute(1, 0).contiguous():
            one_hot = torch.nn.functional.one_hot(
                el, num_classes=values.shape[0]
            ).bool()
            one_hot_combined = torch.logical_or(one_hot_combined, one_hot)
        one_hot_combined[:, torch.where(values == -1)[0][0]] = 0
        intersection = one_hot_combined.long() @ one_hot_combined.T.long()
        one_hot_combined_sum = one_hot_combined.sum(dim=1)
        union = (
            one_hot_combined_sum[:, None]
            + one_hot_combined_sum[None, :]
            - intersection
        )
    else:
        x_elem = x.numel()
        values, inverse = torch.cat([x.view(-1), other.view(-1)]).unique(
            return_inverse=True
        )
        inverse_x = inverse[:x_elem].view_as(x)
        inverse_other = inverse[x_elem:].view_as(other)
        ind_m1 = torch.where(values == -1)[0][0]
        one_hot_x = torch.nn.functional.one_hot(
            inverse_x, num_classes=values.shape[0]
        )
        one_hot_x_combined = one_hot_x.sum(dim=1) >= 1
        one_hot_x_combined[:, ind_m1] = 0
        one_hot_other = torch.nn.functional.one_hot(
            inverse_other, num_classes=values.shape[0]
        )
        one_hot_other_combined = one_hot_other.sum(dim=1) >= 1
        one_hot_other_combined[:, ind_m1] = 0
        intersection = (
            one_hot_x_combined.long() @ one_hot_other_combined.long().T
        )
        union = (
            one_hot_x_combined[:, None, :] + one_hot_other_combined[None, :, :]
            >= 1
        ).sum(dim=-1)
    iou = intersection / union
    return iou


class MutualkNN(DistortionMetric):
    """
    Mutual k-Nearest Neighbors (k-NN) distortion metric.

    This metric identifies mutual k-nearest neighbors between two sets of features
    (or within a single set) and computes how many of the neighbors intersect.
    The similarity can be based on inner product for floating-point features or
    Jaccard index for integer-based (e.g., label sets) features.

    Attributes:
        loss: The loss function to be used, defaults to NegativeInnerProduct.
        k: The number of nearest neighbors to consider. Defaults to 1.
    """

    loss: Loss = NegativeInnerProduct()

    def __init__(
        self,
        k: int = 1,
    ):
        """
        Initialize the MutualkNN metric.

        Args:
            k: The number of nearest neighbors to consider. Defaults to 1.
        """
        super().__init__()
        self.k = k

    def kernel(
        self,
        x: Tensor,
        other: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the kernel matrix based on mutual k-NN.

        If `x` contains floating-point numbers, similarity is computed using
        the inner product. If `x` contains integers, similarity is computed
        using the Jaccard index.

        The diagonal of the similarity matrix is filled with a small negative
        value (or -1 for Jaccard) to prevent an item from being its own neighbor
        unless it's the only option.

        Args:
            x: A 2D tensor of features (N, D) or labels (N, L).
            other: An optional 2D tensor of features (M, D) or labels (M, L).
                   If None, `other` is set to `x`.

        Returns:
            A 2D tensor representing the scaled one-hot encoding of mutual k-NN.
        """
        if x.is_floating_point():
            if other is None:
                other = x
            similarity = x @ other.T
            eps = torch.finfo(x.dtype).eps
        else:
            similarity = jaccard(x, other)
            eps = -1
        indices = similarity.fill_diagonal_(-eps).topk(self.k, dim=1).indices
        one_hot = torch.zeros_like(similarity).scatter_(1, indices, 1)
        one_hot_scaled = one_hot / (sqrt(one_hot.shape[0] * self.k))
        return one_hot_scaled

    kernel_source = kernel
    kernel_target = kernel

    def __str__(self):
        """Return a string representation of the MutualkNN metric."""
        return f"Mutual k-NN (k={self.k})"
