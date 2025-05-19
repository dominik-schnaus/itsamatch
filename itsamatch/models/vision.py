"""Vision models for image feature extraction."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Callable, override

import clip
import timm
import torch
import torchvision
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class VisionModel(Module, ABC):
    """Abstract base class for vision models."""

    def __init__(self, name: str, resolution: int | None = None):
        """
        Initialize the VisionModel.

        Args:
            name: Name of the model. Can include resolution like "model_name@resolution".
            resolution: Image resolution. Defaults to 224 if not specified or in name.
        """
        super().__init__()
        self.name = name

        if "@" in name and resolution is None:
            split_name = name.split("@")
            assert len(split_name) == 2, f"Invalid name: {name}"
            name, resolution_str = split_name
            assert resolution_str.isdigit(), f"Invalid name: {name}"
            resolution = int(resolution_str)
        if resolution is None:
            resolution = 224
        self.resolution = resolution

        self.model_name = name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.float32

    @cached_property
    def model(self) -> Module:
        """Lazy‐load the actual backbone."""
        return self._get_model().to(device=self.device, dtype=self.dtype).eval()

    @cached_property
    def transform(self) -> Callable:
        """Lazy‐build the per‐model transform pipeline."""
        return self._get_transform()

    def get_batch_size(self, gpu_memory: int) -> int:
        """
        Get the batch size of the model.

        This could be more complex depending on the model and the GPU
        memory. For now, we just return a fixed value.

        Args:
            gpu_memory: GPU memory in Byte.

        Returns:
            Batch size.
        """
        return 16

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_model(self) -> Module:
        """
        Returns a ready‐to‐use nn.Module.

        Returns:
            The neural network model.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_transform(self) -> Callable:
        """
        Returns a Callable transform.

        Returns:
            A callable transform for preprocessing images.
        """
        raise NotImplementedError

    @override
    def to(self, *args: Any, **kwargs: Any) -> "VisionModel":
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

    def __str__(self):
        """Return a string representation of the model."""
        return self.__class__.__name__.lower() + "_" + self.name.lower()


class DINOv2(VisionModel):
    """DINOv2 vision model."""

    _repo: str = "facebookresearch/dinov2"

    def __init__(self, name: str, resolution: int | None = None):
        """
        Initialize the DINOv2 model.

        Args:
            name: Name of the DINOv2 model variant (e.g., "vit_small", "vit_base").
            resolution: Image resolution.
        """
        super().__init__(name, resolution)

        dinov2_name = self.model_name.replace("/", "").replace("-", "").lower()
        self._model_name = f"dinov2_{dinov2_name}"

    def _get_model(self) -> Module:
        """Load the DINOv2 model from torch.hub."""
        return torch.hub.load(self._repo, self._model_name, pretrained=True)

    def _get_transform(self) -> Callable:
        """Get the image transformation pipeline for DINOv2."""
        return Compose(
            [
                Resize((self.resolution, self.resolution), antialias=True),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the DINOv2 model."""
        return self.model(x)


class DeiT(VisionModel):
    """DeiT (Data-efficient Image Transformer) vision model."""

    _repo: str = "facebookresearch/deit:main"
    _character_to_size: dict[str, str] = {
        "T": "tiny",
        "S": "small",
        "B": "base",
    }

    def __init__(self, name: str, resolution: int | None = None):
        """
        Initialize the DeiT model.

        Args:
            name: Name of the DeiT model variant (e.g., "DeiT-T/16", "DeiT-B/16d@384").
            resolution: Image resolution.
        """
        super().__init__(name, resolution)

        assert self.model_name.startswith("DeiT-"), (
            f"Invalid name: {self.model_name}"
        )
        deit_name = self.model_name[5:]

        deit_name_list = ["deit"]
        size_char = deit_name.split("/")[0]
        assert size_char in self._character_to_size, (
            f"Invalid name: {self.model_name}"
        )
        deit_name_list.append(self._character_to_size[size_char])

        patch_size = deit_name.split("/")[1]
        if patch_size.endswith("d"):
            deit_name_list.append("distilled")
            patch_size = patch_size[:-1]

        assert patch_size.isdigit(), f"Invalid name: {self.model_name}"
        deit_name_list.append(f"patch{patch_size}")

        deit_name_list.append(str(self.resolution))

        self._model_name = "_".join(deit_name_list)

    def _get_model(self) -> Module:
        """Load the DeiT model from torch.hub."""
        return torch.hub.load(self._repo, self._model_name, pretrained=True)

    def _get_transform(self) -> Callable:
        """Get the image transformation pipeline for DeiT."""
        return Compose(
            [
                Resize(
                    (self.resolution, self.resolution),
                    interpolation=3,
                    antialias=True,
                ),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the DeiT model."""
        return self.model(x)


class CLIP(VisionModel):
    """CLIP (Contrastive Language-Image Pre-Training) vision model."""

    def __init__(self, name: str, resolution: int | None = None):
        """
        Initialize the CLIP model.

        Args:
            name: Name of the CLIP model variant (e.g., "ViT-B/32", "RN50").
            resolution: Image resolution (224 or 336).
        """
        super().__init__(name, resolution)

        if self.resolution == 336:
            self.model_name = f"{self.model_name}@336px"
        else:
            assert self.resolution == 224, (
                f"Invalid resolution: {self.resolution}"
            )

    def _get_model(self):
        """Load the CLIP model."""
        model = clip.load(self.model_name, device=self.device)[0]
        return model

    def _get_transform(self):
        """Get the image transformation pipeline for CLIP."""
        return clip.load(self.model_name, device="cpu")[1]

    def forward(self, x):
        """Forward pass through the CLIP image encoder."""
        return self.model.encode_image(x)


class ConvNeXt(VisionModel):
    """ConvNeXt vision model."""

    _character_to_size: dict[str, str] = {
        "B": "base",
        "L": "large",
        "XL": "xlarge",
    }

    def __init__(
        self, name: str, resolution: int | None = None, use_head: bool = True
    ):
        """
        Initialize the ConvNeXt model.

        Args:
            name: Name of the ConvNeXt model variant (e.g., "CN-B-1", "CN-L-22ft@384").
            resolution: Image resolution.
        """
        super().__init__(name, resolution)
        self.use_head = use_head

        assert self.model_name.startswith("CN-"), (
            f"Invalid name: {self.model_name}"
        )
        cn_name = self.model_name[3:]

        cn_name_list = ["convnext"]
        size_char = cn_name.split("-")[0]
        assert size_char in self._character_to_size, (
            f"Invalid name: {self.model_name}"
        )
        cn_name_list.append(self._character_to_size[size_char] + ".fb")

        dataset = cn_name.split("-")[-1]

        if dataset.startswith("1"):
            cn_name_list.append("in1k")
        elif dataset.startswith("22"):
            cn_name_list.append("in22k")

        if dataset.endswith("ft"):
            cn_name_list.append("ft_in1k")

        if self.resolution == 384:
            cn_name_list.append(str(self.resolution))

        self._model_name = "_".join(cn_name_list)

    def _get_model(self):
        """Load the ConvNeXt model usi‚ng timm."""
        model = timm.create_model(self._model_name, pretrained=True)
        return model

    def _get_transform(self):
        """Get the image transformation pipeline for ConvNeXt."""
        data_config = timm.data.resolve_model_data_config(self.model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        return transform

    def forward(self, x):
        """Forward pass through the ConvNeXt model."""
        output = self.model.forward_features(x)
        if self.use_head:
            output = self.model.forward_head(output, pre_logits=True)
        return output


class DINO(VisionModel):
    """DINO (self-DIstillation with NO labels) vision model."""

    _repo: str = "facebookresearch/dino:main"

    def __init__(self, name: str, resolution: int | None = None):
        """
        Initialize the DINO model.

        Args:
            name: Name of the DINO model variant (e.g., "ViT-B/8", "RN50").
            resolution: Image resolution.
        """
        super().__init__(name, resolution)

        dino_name = self.model_name.replace("/", "").replace("-", "").lower()
        if dino_name == "rn50":
            dino_name = "resnet50"
        self._model_name = f"dino_{dino_name}"

    def _get_model(self) -> Module:
        """Load the DINO model from torch.hub."""
        return torch.hub.load(self._repo, self._model_name, pretrained=True)

    def _get_transform(self) -> Callable:
        """Get the image transformation pipeline for DINO."""
        return Compose(
            [
                Resize(
                    (self.resolution, self.resolution),
                    interpolation=3,
                    antialias=True,
                ),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the DINO model."""
        return self.model(x)


class RandomViT(VisionModel):
    """Vision Transformer (ViT) model with random weights."""

    class_name: str = "Random ViT"

    def __init__(self, name: str, seed: int, resolution: int | None = None):
        """
        Initialize the RandomViT model.

        Args:
            name: Name of the ViT architecture (e.g., "ViT-H/14", "ViT-B/8").
            seed: Random seed for weight initialization.
            resolution: Image resolution.
        """
        super().__init__(name=name, resolution=resolution)
        self.seed = seed

        self._model_name = (
            self.model_name.replace("/", "_").replace("-", "_").lower()
        )

        self.__class__.__name__ = self.class_name

    def _get_model(self) -> Module:
        """Initialize a ViT model with random weights using a specific seed."""
        initial_seed = torch.initial_seed()
        torch.manual_seed(self.seed)
        model = getattr(
            torchvision.models.vision_transformer, self._model_name
        )(weights=None)
        torch.manual_seed(initial_seed)
        return model

    def _get_transform(self) -> Callable:
        """Get the image transformation pipeline for RandomViT."""
        transform = Compose(
            [
                Resize(
                    (self.resolution, self.resolution),
                    interpolation=3,
                    antialias=True,
                ),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        return transform

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the RandomViT model."""
        # Reshape and permute the input tensor
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return x

    def __str__(self):
        """Return a string representation of the RandomViT model including the seed."""
        self.__class__.__name__ = self.class_name
        return super().__str__() + f"_{self.seed}"


class PixelValues(VisionModel):
    """A 'model' that directly uses normalized pixel values as features."""

    class_name: str = "Pixel values"

    def __init__(self, resolution: int | None = None):
        """
        Initialize the PixelValues model.

        Args:
            resolution: Image resolution to which images will be resized.
        """
        super().__init__(name="", resolution=resolution)

        self.__class__.__name__ = self.class_name

    def _get_model(self) -> Module:
        """Return an identity model as no actual model is needed."""
        model = torch.nn.Identity()
        return model

    def _get_transform(self) -> Callable:
        """Get the image transformation pipeline for PixelValues."""
        transform = Compose(
            [
                Resize((self.resolution, self.resolution), antialias=True),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        return transform

    def forward(self, x: Tensor) -> Tensor:
        """Flatten the input tensor to represent pixel values."""
        B = x.shape[0]
        return x.view(B, -1)

    def __str__(self):
        """Return a string representation of the PixelValues model including the resolution."""
        self.__class__.__name__ = self.class_name
        return f"pixel_values_{self.resolution}"


class Random(VisionModel):
    """A 'model' that generates random features."""

    def __init__(self, dim: int, seed: int):
        """
        Initialize the Random model.

        Args:
            dim: Dimensionality of the random feature vectors.
            seed: Random seed for generating features.
        """
        super().__init__(name="", resolution=None)
        self.dim = dim
        self.seed = seed

    def _get_model(self) -> Module:
        """Return an identity model as no actual model is needed for feature generation."""
        model = torch.nn.Identity()
        return model

    def _get_transform(self) -> Callable:
        """Get a minimal image transformation pipeline."""
        return Compose(
            [
                Resize((1, 1), antialias=True),
                ToTensor(),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Generate random features for the input batch."""
        B = x.shape[0]
        return torch.randn(B, self.dim, device=x.device, dtype=x.dtype)

    def __str__(self):
        """Return a string representation of the Random model including dimension and seed."""
        return f"random_{self.dim}_{self.seed}"
