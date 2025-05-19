"""Inner product distortion metric."""

from .gw import GromovWasserstein
from .losses import Loss, NegativeInnerProduct


class InnerProduct(GromovWasserstein):
    """Inner product distortion metric.

    This metric uses the negative inner product as the loss function and
    the pairwise Euclidean distance matrix as the kernel.
    """

    loss: Loss = NegativeInnerProduct()

    def __str__(self):
        """Return the string representation of the InnerProduct metric."""
        return "InnerProduct"
