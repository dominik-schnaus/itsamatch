"""Defines various loss functions for comparing kernel matrices in QAP.

Each loss function provides methods to compute the full cost matrix and
its components (h1, h2, f1, f2) for factorized solvers.
"""

import torch
from torch import Tensor


class Loss:
    """Abstract base class for loss functions.

    A loss function defines how to compute the cost between two kernel matrices
    or their elements. It also provides components for factorized QAP solvers.

    It needs to be decomposable into :math:`l(x, y) = f_1(x) + f_2(y) - h_1(x)h_2(y)`.

    For more details, see Eq. 9 of our paper:
    "It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data"
    (https://arxiv.org/pdf/2503.24129)
    or Eq. 5 of the original paper on Gromov-Wasserstein distance:
    "Gromov-Wasserstein Averaging of Kernel and Distance Matrices"
    (https://proceedings.mlr.press/v48/peyre16.pdf)
    """

    @staticmethod
    def __call__(x: Tensor, y: Tensor) -> Tensor:
        """Compute the element-wise loss between two tensors.

        Args:
            x: The first input tensor.
            y: The second input tensor.

        Returns:
            A tensor containing the element-wise loss.
        """
        raise NotImplementedError

    @staticmethod
    def h1(x: Tensor) -> Tensor:
        """Compute the h1 component for factorized solvers.

        Args:
            x: The input tensor (typically from the source kernel matrix).

        Returns:
            The h1 component tensor.
        """
        raise NotImplementedError

    @staticmethod
    def h2(y: Tensor) -> Tensor:
        """Compute the h2 component for factorized solvers.

        Args:
            y: The input tensor (typically from the target kernel matrix).

        Returns:
            The h2 component tensor.
        """
        raise NotImplementedError

    @staticmethod
    def f1(x: Tensor) -> Tensor:
        """Compute the f1 component for factorized solvers (constant term part).

        Args:
            x: The input tensor (typically from the source kernel matrix).

        Returns:
            The f1 component tensor.
        """
        raise NotImplementedError

    @staticmethod
    def f2(y: Tensor) -> Tensor:
        """Compute the f2 component for factorized solvers (constant term part).

        Args:
            y: The input tensor (typically from the target kernel matrix).

        Returns:
            The f2 component tensor.
        """
        raise NotImplementedError

    @classmethod
    def __repr__(cls):
        """Return the string representation of the loss class."""
        return cls.__name__


class SquaredLoss(Loss):
    """Squared L2 loss: (x - y)^2."""

    @staticmethod
    def __call__(x: Tensor, y: Tensor) -> Tensor:
        """Compute the element-wise squared L2 loss.

        Args:
            x: The first input tensor.
            y: The second input tensor.

        Returns:
            A tensor containing (x - y)^2.
        """
        return (x - y).pow(2)

    @staticmethod
    def h1(x: Tensor) -> Tensor:
        """Compute h1 = x for squared loss."""
        return x

    @staticmethod
    def h2(y: Tensor) -> Tensor:
        """Compute h2 = 2y for squared loss."""
        return 2 * y

    @staticmethod
    def f1(x: Tensor) -> Tensor:
        """Compute f1 = x^2 for squared loss."""
        return x**2

    @staticmethod
    def f2(y: Tensor) -> Tensor:
        """Compute f2 = y^2 for squared loss."""
        return y**2


class NegativeInnerProduct(Loss):
    """Negative inner product loss: -x * y."""

    @staticmethod
    def __call__(x: Tensor, y: Tensor) -> Tensor:
        """Compute the element-wise negative inner product.

        Args:
            x: The first input tensor.
            y: The second input tensor.

        Returns:
            A tensor containing -x * y.
        """
        return -x * y

    @staticmethod
    def h1(x: Tensor) -> Tensor:
        """Compute h1 = x for negative inner product loss."""
        return x

    @staticmethod
    def h2(y: Tensor) -> Tensor:
        """Compute h2 = y for negative inner product loss."""
        return y

    @staticmethod
    def f1(x: Tensor) -> Tensor:
        """Compute f1 = 0 for negative inner product loss."""
        return torch.zeros_like(x)

    @staticmethod
    def f2(y: Tensor) -> Tensor:
        """Compute f2 = 0 for negative inner product loss."""
        return torch.zeros_like(y)


class KL(Loss):
    """Kullback-Leibler (KL) divergence: x * log(x/y) - x + y."""

    @staticmethod
    def __call__(x: Tensor, y: Tensor) -> Tensor:
        """Compute the element-wise KL divergence.

        Args:
            x: The first input tensor (P).
            y: The second input tensor (Q).

        Returns:
            A tensor containing KL(P || Q).
        """
        return x * (x / y).log() - x + y

    @staticmethod
    def h1(x: Tensor) -> Tensor:
        """Compute h1 = x for KL divergence."""
        return x

    @staticmethod
    def h2(y: Tensor) -> Tensor:
        """Compute h2 = log(y) for KL divergence."""
        return y.log()

    @staticmethod
    def f1(x: Tensor) -> Tensor:
        """Compute f1 = x * log(x) - x for KL divergence."""
        return x * x.log() - x

    @staticmethod
    def f2(y: Tensor) -> Tensor:
        """Compute f2 = y for KL divergence."""
        return y
