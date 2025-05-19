"""Abstract base class for distortion metrics."""

from abc import ABC, abstractmethod

from torch import Tensor

from .losses import Loss


class DistortionMetric(ABC):
    """Abstract base class for distortion metrics.

    A distortion metric defines how to compute kernel matrices for source and target
    modalities, and specifies a loss function between the kernel matrices.

    It can be used to measure the similarity between two sets of representations:

    Example:
        >>> # Instantiate a distortion metric
        >>> metric = # YourDistortionMetricClass
        >>>
        >>> # Create dummy source and target tensors (e.g., embeddings)
        >>> source_embeddings = torch.randn(10, 128) # 10 samples, 128 dimensions
        >>> target_embeddings = torch.randn(10, 128) # 10 samples, 128 dimensions
        >>>
        >>> # Compute kernel matrices
        >>> source_kernel = metric.kernel_source(source_features)
        >>> target_kernel = metric.kernel_target(target_features)
        >>>
        >>> # source_kernel and target_kernel can now be used with a loss function
        >>> # For example, to compute a matching cost:
        >>> cost_matrix = metric.loss(source_kernel, target_kernel)
        >>>
        >>> # One can also get the factorization using the f1, f2, h1, and h2 methods of the loss
    """

    loss: Loss

    @abstractmethod
    def kernel_source(
        self,
        x: Tensor,
        other: Tensor | None = None,
    ) -> Tensor:
        """Compute the kernel matrix for the source modality.

        Args:
            x: The first input tensor for the source modality.
            other: The second input tensor for the source modality. If None, x is used.

        Returns:
            The kernel matrix for the source modality.
        """
        raise NotImplementedError

    @abstractmethod
    def kernel_target(
        self,
        x: Tensor,
        other: Tensor | None = None,
    ) -> Tensor:
        """Compute the kernel matrix for the target modality.

        Args:
            x: The first input tensor for the target modality.
            other: The second input tensor for the target modality. If None, x is used.

        Returns:
            The kernel matrix for the target modality.
        """
        raise NotImplementedError

    def __repr__(self):
        """Return a string representation of the DistortionMetric instance."""
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])})"
