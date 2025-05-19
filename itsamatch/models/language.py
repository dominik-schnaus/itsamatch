"""This module defines abstract and concrete language models for text embedding."""

import os
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, override

import clip
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.nn import Module


class LanguageModel(Module, ABC):
    """Abstract base class for language models."""

    def __init__(self, name: str = None):
        """
        Initialize the LanguageModel.

        Args:
            name: The name of the language model.
        """
        super().__init__()
        self.name = name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.float32

    @cached_property
    def model(self) -> Module:
        """Lazy‐load the actual backbone."""
        return self._get_model().to(self.device)

    @abstractmethod
    def _get_model(self):
        """Abstract method to load and return the underlying model."""
        raise NotImplementedError

    @override
    def to(self, *args: Any, **kwargs: Any) -> "LanguageModel":
        """
        Move the model to the specified device.

        Args:
            device: The target device.
            *args: Additional arguments for super().to().
            **kwargs: Additional keyword arguments for super().to().

        Returns:
            The model itself.
        """
        """See :meth:`torch.nn.Module.to`."""
        # this converts `str` device to `torch.device`
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self.device = device

        if dtype is not None:
            self.dtype = dtype
        return super().to(*args, **kwargs)

    def get_batch_size(self, gpu_memory: int) -> int:
        """
        Get the recommended batch size based on GPU memory.

        Args:
            gpu_memory: The available GPU memory in GB.

        Returns:
            The recommended batch size.
        """
        return 16

    @abstractmethod
    def forward(self, x: list[str]) -> Tensor:
        """
        Abstract method for the forward pass of the model.

        Args:
            x: List of strings to be tokenized and embedded.

        Returns:
            Output tensor.
        """
        raise NotImplementedError

    def __repr__(self):
        """Return a string representation of the model."""
        return self.__class__.__name__.lower() + "_" + self.name.lower()


class SentenceT(LanguageModel):
    """Language model using SentenceTransformer models."""

    def __init__(self, name: str = None):
        """
        Initialize the SentenceT model.

        Args:
            name: The name of the SentenceTransformer model.
        """
        super().__init__(name)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _get_model(self):
        """Load and return the SentenceTransformer model."""
        model = SentenceTransformer(self.name)
        return model

    def forward(self, x: list[str]) -> Tensor:
        """
        Perform a forward pass to get sentence embeddings.

        Args:
            x: List of strings to be tokenized and embedded.

        Returns:
            Tensor containing sentence embeddings.
        """
        features = self.model.tokenize(x)
        features = {
            key: value.to(device=self.device) for key, value in features.items()
        }
        return self.model.forward(features)["sentence_embedding"]


class CLIP(LanguageModel):
    """Language model using the CLIP language models."""

    def _get_model(self):
        """Load and return the CLIP model."""
        model, _ = clip.load(self.name, device=self.device)
        return model

    def forward(self, x: list[str]) -> Tensor:
        """
        Perform a forward pass to get text embeddings using CLIP.

        Args:
            x: List of strings to be tokenized and embedded.

        Returns:
            Tensor containing text embeddings.
        """
        x = clip.tokenize(x, truncate=True).to(device=self.device)
        return torch.as_tensor(self.model.encode_text(x))

    def __repr__(self):
        """Return a string representation of the CLIP model."""
        return self.__class__.__name__.lower() + "_" + self.name.lower()
