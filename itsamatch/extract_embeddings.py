"""
This module provides functionalities for extracting, managing, and processing
vision and language embeddings from datasets using specified models.
It includes utilities for computing embeddings, submitting jobs for computation,
loading pre-computed embeddings, and preparing them for further use.
"""

import gc
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import torch
from submitit.helpers import Checkpointable
from torch.nn import Module
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets.vision import StandardTransform
from tqdm import tqdm, trange

from itsamatch.datasets import Dataset
from itsamatch.models.language import LanguageModel
from itsamatch.models.vision import VisionModel
from itsamatch.utils import (
    monitor_jobs,
    submit_grouped,
)


def get_hash(thing):
    """
    Generates a stable MD5 hash for any JSON-serializable Python object.

    From https://death.andgravity.com/stable-hashing

    Args:
        thing: The Python object to hash.

    Returns:
        A hexadecimal string representing the MD5 hash.
    """
    return (
        hashlib.md5(
            json.dumps(
                thing,
                ensure_ascii=False,
                sort_keys=True,
                indent=None,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        .digest()
        .hex()
    )


def encode_prompts(prompts):
    """
    Encodes a list of prompts into a stable hash.

    Args:
        prompts: A list of prompt strings.

    Returns:
        A hexadecimal string representing the MD5 hash of the prompts,
        or None if prompts is None.
    """
    if prompts is None:
        return None
    return get_hash(prompts)


def get_vision_embedding_path(
    root: str,
    model: Module,
    dataset: Dataset,
) -> Path:
    """
    Constructs the path for storing/retrieving vision embeddings.

    Args:
        root: The root directory for embeddings.
        model: The vision model used.
        dataset: The dataset for which embeddings are generated.

    Returns:
        A Path object pointing to the vision embedding file.
    """
    model_name = str(model).replace("/", "").replace(" ", "-").lower()
    return root / str(dataset) / "vision" / f"{model_name}.pt"


def get_language_embedding_path(
    root: str,
    model: Module,
    dataset: Dataset,
) -> Path:
    """
    Constructs the path for storing/retrieving language embeddings.

    Args:
        root: The root directory for embeddings.
        model: The language model used.
        dataset: The dataset for which embeddings are generated.

    Returns:
        A Path object pointing to the language embedding file.
    """
    model_name = str(model).replace("/", "").lower()
    language_name = (
        f"language_{encode_prompts(dataset.prompts)}"
        if dataset.prompts
        else "language"
    )
    return root / str(dataset) / language_name / f"{model_name}.pt"


def get_labels_path(
    root: str,
    dataset: Dataset,
) -> Path:
    """
    Constructs the path for storing/retrieving dataset labels.

    Args:
        root: The root directory for embeddings.
        dataset: The dataset for which labels are generated.

    Returns:
        A Path object pointing to the labels file.
    """
    return root / str(dataset) / "labels.pt"


def load_embeddings(path: Path, mmap: bool | None = None) -> torch.Tensor:
    """
    Loads embeddings from a specified path.

    Args:
        path: The Path object pointing to the embedding file.
        mmap: Whether to use memory-mapping. Defaults to None.

    Returns:
        A torch.Tensor containing the loaded embeddings.

    Raises:
        FileNotFoundError: If the embedding file does not exist.
    """
    if path.exists():
        print(f"Loading embeddings from {path}")
        return torch.load(path, mmap=mmap)
    else:
        print(f"Files available in {path.parent}:")
        for file in path.parent.iterdir():
            print(file)
        raise FileNotFoundError(f"File not found: {path}")


def compute_missing_embeddings(
    path_to_embeddings: Path,
    vision_models: list[Module],
    language_models: list[Module],
    datasets: list[Dataset],
) -> None:
    """
    Computes and saves embeddings that are not yet present in the specified directory.

    This function iterates through datasets and models, checks for existing
    embedding files, and computes them if they are missing.

    Args:
        path_to_embeddings: The root directory where embeddings are stored.
        vision_models: A list of vision models to use.
        language_models: A list of language models to use.
        datasets: A list of datasets to process.
    """
    for dataset in datasets:
        # Compute vision embeddings if missing
        for vision_model in vision_models:
            vision_embedding_path = get_vision_embedding_path(
                path_to_embeddings, vision_model, dataset
            )
            if not vision_embedding_path.exists():
                VisionEmbeddingComputer()(
                    vision_embedding_path,
                    model=vision_model,
                    dataset=dataset,
                )
        # Compute language embeddings if missing
        for language_model in language_models:
            language_embedding_path = get_language_embedding_path(
                path_to_embeddings, language_model, dataset
            )
            if not language_embedding_path.exists():
                LanguageEmbeddingComputer()(
                    language_embedding_path,
                    model=language_model,
                    dataset=dataset,
                )
        # Compute labels if missing
        labels_path = get_labels_path(path_to_embeddings, dataset)
        if not labels_path.exists():
            LabelComuputer()(
                labels_path,
                dataset=dataset,
            )


def submit_missing_embeddings(
    path_to_embeddings: Path,
    path_to_logs: Path,
    vision_models: list[Module],
    language_models: list[Module],
    datasets: list[Dataset],
    num_jobs: int,
    runtime_per_function: int,
    cluster: str = "slurm",
    **submitit_kwargs,
) -> None:
    """
    Submits jobs to a cluster (e.g., Slurm) to compute missing embeddings.

    This function identifies missing embeddings and creates job submission tasks
    for them, distinguishing between GPU-intensive (vision, language) and
    CPU-intensive (label) computations.

    It is equivalent to `compute_missing_embeddings`, but uses a job scheduler
    to manage the computation in parallel.

    Args:
        path_to_embeddings: Root directory for embeddings.
        path_to_logs: Directory for storing job logs.
        vision_models: List of vision models.
        language_models: List of language models.
        datasets: List of datasets to process.
        num_jobs: Total number of parallel jobs to submit.
        runtime_per_function: Estimated runtime for each embedding computation task.
        cluster: The cluster management system (e.g., "slurm").
        **submitit_kwargs: Additional arguments for the submitit executor.
    """
    # Executor for GPU tasks
    gpu_jobs = []
    for dataset in datasets:
        # Prepare vision embedding jobs
        for vision_model in vision_models:
            vision_embedding_path = get_vision_embedding_path(
                path_to_embeddings, vision_model, dataset
            )
            if not vision_embedding_path.exists():
                job = partial(
                    VisionEmbeddingComputer(),
                    path=vision_embedding_path,
                    model=vision_model,
                    dataset=dataset,
                )
                gpu_jobs.append(job)
        # Prepare language embedding jobs
        for language_model in language_models:
            language_embedding_path = get_language_embedding_path(
                path_to_embeddings, language_model, dataset
            )
            if not language_embedding_path.exists():
                job = partial(
                    LanguageEmbeddingComputer(),
                    path=language_embedding_path,
                    model=language_model,
                    dataset=dataset,
                )
                gpu_jobs.append(job)

    # Executor for CPU tasks
    cpu_jobs = []
    for dataset in datasets:
        # Prepare label computation jobs
        labels_path = get_labels_path(path_to_embeddings, dataset)
        if not labels_path.exists():
            job = partial(
                LabelComuputer(),
                path=labels_path,
                dataset=dataset,
            )
            cpu_jobs.append(job)

    # If no jobs to run, exit
    if len(gpu_jobs) == 0 and len(cpu_jobs) == 0:
        print("All embeddings already exist.")
        return

    # Submit jobs based on availability and num_jobs configuration
    if len(gpu_jobs) == 0 or len(cpu_jobs) == 0:
        # Submit all jobs to one group if only one type of job exists
        jobs = submit_grouped(
            gpu_jobs + cpu_jobs,
            num_jobs=num_jobs,
            path_to_logs=path_to_logs,
            runtime_per_function=runtime_per_function,
            cluster=cluster,
            **submitit_kwargs,
        )
    elif num_jobs > 1:
        # Distribute jobs between GPU and CPU if multiple jobs are allowed
        gpu_jobs = submit_grouped(
            gpu_jobs,
            num_jobs=num_jobs - 1,  # Allocate most jobs to GPU tasks
            path_to_logs=path_to_logs,
            runtime_per_function=runtime_per_function,
            cluster=cluster,
            **submitit_kwargs,
        )
        submitit_kwargs_cpu = submitit_kwargs.copy()
        submitit_kwargs_cpu["gpus_per_node"] = (
            0  # Ensure CPU jobs don't request GPUs
        )
        cpu_jobs = submit_grouped(
            cpu_jobs,
            num_jobs=1,  # Allocate one job for CPU tasks
            path_to_logs=path_to_logs,
            runtime_per_function=runtime_per_function,
            cluster=cluster,
            **submitit_kwargs_cpu,
        )
        jobs = gpu_jobs + cpu_jobs
    else:
        # Submit all jobs to one group if only one job is allowed
        jobs = submit_grouped(
            gpu_jobs + cpu_jobs,
            num_jobs=1,
            path_to_logs=path_to_logs,
            runtime_per_function=runtime_per_function,
            cluster=cluster,
            **submitit_kwargs,
        )

    # Wait for all jobs to finish
    monitor_jobs(jobs)
    return


def load_and_preprocess_embeddings(
    vision_model: VisionModel,
    language_model: LanguageModel,
    dataset: Dataset,
    path_to_embeddings: Path,
    normalize_before_averaging: bool = False,
):
    """
    Loads vision and language embeddings and labels, then preprocesses them.

    Preprocessing includes optional normalization before averaging (for vision
    embeddings) and averaging of language embeddings if they have an extra
    dimension (e.g., from multiple prompts). Language embeddings are always
    normalized after potential averaging.

    Args:
        vision_model: The vision model used for embeddings.
        language_model: The language model used for embeddings.
        dataset: The dataset for which embeddings were generated.
        path_to_embeddings: Root directory where embeddings are stored.
        normalize_before_averaging: If True, normalize vision and language
                                    embeddings before any averaging.

    Returns:
        vision_embeddings (torch.Tensor): Processed vision embeddings.
        language_embeddings (torch.Tensor): Processed language embeddings.
        labels (torch.Tensor): Dataset labels.
    """
    # load embeddings
    vision_embeddings = load_embeddings(
        get_vision_embedding_path(
            root=path_to_embeddings,
            model=vision_model,
            dataset=dataset,
        )
    )
    language_embeddings = load_embeddings(
        get_language_embedding_path(
            root=path_to_embeddings,
            model=language_model,
            dataset=dataset,
        )
    )
    labels = load_embeddings(
        get_labels_path(
            root=path_to_embeddings,
            dataset=dataset,
        )
    )

    # normalize vision embeddings before averaging
    if normalize_before_averaging:
        vision_embeddings = normalize(vision_embeddings, dim=-1)
        language_embeddings = normalize(language_embeddings, dim=-1)

    # average language embeddings if they are from multiple prompts (ndim == 3)
    if language_embeddings.ndim == 3:
        language_embeddings = language_embeddings.nanmean(dim=1)

    # normalize language embeddings (after potential averaging)
    language_embeddings = normalize(language_embeddings, dim=-1)

    return vision_embeddings, language_embeddings, labels


def clear_memory():
    """Clears GPU cache and runs Python's garbage collector."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    gc.collect()


class VisionEmbeddingComputer(Checkpointable):
    """
    Computes vision embeddings for a given dataset using a specified vision model.
    Inherits from Checkpointable to support saving and resuming computation.
    """

    embedding_list: list[torch.Tensor] = []

    @torch.inference_mode()
    def __call__(
        self,
        path: Path,
        model: VisionModel,
        dataset: Dataset,
        gpu_memory: int = 16,
        num_workers: int = 4,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.double,
        checkpoint_path: Path | None = None,
    ):
        """
        Computes and saves vision embeddings.

        Args:
            path: Path to save the computed embeddings.
            model: The VisionModel instance to use.
            dataset: The Dataset instance to process.
            gpu_memory: Estimated GPU memory (in GB) to adjust batch size.
            num_workers: Number of worker processes for data loading.
            device: The torch.device to use for computation. Auto-detects CUDA if None.
            dtype: The torch.dtype for embeddings.
            checkpoint_path: Path to a checkpoint file to resume from.
        """
        print(f"Computing embeddings for {dataset} with {model}")
        # Determine batch size based on model and GPU memory
        batch_size = model.get_batch_size(gpu_memory)

        # Set dataset transforms
        dataset.transform = model.transform
        dataset.transforms = StandardTransform(model.transform, None)

        # Define a collate function to extract image tensors
        def vision_collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
            return default_collate([b[0] for b in batch])

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=vision_collate_fn,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Move model to device and set to evaluation mode
        model = model.to(device=device).eval()

        # Load from checkpoint if provided
        if checkpoint_path is not None and checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            self.embedding_list = torch.load(checkpoint_path)
        else:
            self.embedding_list = []

        # Process batches
        for i, batch in enumerate(tqdm(dataloader)):
            if i < len(
                self.embedding_list
            ):  # Skip already processed batches if resuming
                continue
            batch = batch.to(device=device)
            out = model(batch)  # Compute embeddings
            out = out.cpu().to(dtype=dtype)  # Move to CPU before storing
            self.embedding_list.append(out)

        # Concatenate all batch embeddings
        embeddings = torch.cat(self.embedding_list, dim=0)
        print(f"Saving embeddings to {path}")
        os.makedirs(path.parent, exist_ok=True)
        torch.save(embeddings, path)

        # Clean up
        del embeddings, batch, model, dataloader
        self.embedding_list = []
        clear_memory()
        return

    def checkpoint(self, *args, **kwargs):
        """
        Saves the current list of computed embeddings as an intermediate checkpoint.
        This method is typically called by a job scheduler like submitit.

        Args:
            *args: Positional arguments, path is expected as the first argument.
            **kwargs: Keyword arguments, path can also be passed as a keyword.

        Returns:
            The result of the superclass's checkpoint method.
        """
        # Determine the path for the checkpoint file
        path = kwargs.get("path", args[0])
        if isinstance(path, str):
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        checkpoint_path = path.parent / f"intermediate_{path.name}"
        torch.save(self.embedding_list, checkpoint_path)
        print(f"Saving intermediate checkpoint to {checkpoint_path}")

        # Pass the checkpoint_path to the next invocation of __call__
        kwargs["checkpoint_path"] = checkpoint_path
        return super().checkpoint(*args, **kwargs)


class ClassDataset(Dataset):
    """
    A simple dataset wrapper for a list of class names and optional text prompts.
    Used primarily for generating language embeddings for class labels.
    """

    def __init__(self, classes, prompts):
        """
        Initializes the ClassDataset.

        Args:
            classes: A list of class names (strings).
            prompts: A list of prompt templates (strings with '{}' for class name)
                     or None if no prompting is used.
        """
        self.classes = classes
        self.prompts = prompts

    def __len__(self):
        """Returns the number of classes."""
        return len(self.classes)

    def __getitem__(self, idx):
        """
        Gets the item(s) at the given index.

        If prompts are provided, it returns a list of formatted prompt strings
        for the class at `idx`. Otherwise, it returns a list containing
        just the class name.

        Args:
            idx: The index of the class.

        Returns:
            A list of strings.
        """
        if self.prompts is None:
            return [self.classes[idx]]
        return [prompt.format(self.classes[idx]) for prompt in self.prompts]


class LanguageEmbeddingComputer(Checkpointable):
    """
    Computes language embeddings for a given dataset using a specified language model.
    Handles both paired datasets (e.g., image-caption pairs) and class-based datasets.
    Inherits from Checkpointable to support saving and resuming computation.
    """

    embedding_list: list[torch.Tensor] = []

    @torch.inference_mode()
    def __call__(
        self,
        path: Path,
        model: LanguageModel | None,
        dataset: Dataset | None,
        gpu_memory: int = 16,
        num_workers: int = 4,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.double,
        checkpoint_path: Path | None = None,
    ):
        """
        Computes and saves language embeddings.

        Args:
            path: Path to save the computed embeddings.
            model: The LanguageModel instance to use.
            dataset: The Dataset instance to process.
            gpu_memory: Estimated GPU memory (in GB) to adjust batch size.
            num_workers: Number of worker processes for data loading.
            device: The torch.device to use for computation. Auto-detects CUDA if None.
            dtype: The torch.dtype for embeddings.
            checkpoint_path: Path to a checkpoint file to resume from.
        """
        print(f"Computing embeddings for {dataset} with {model}")
        # Determine batch size
        batch_size = model.get_batch_size(gpu_memory)

        # Configure DataLoader based on dataset type (paired or class-based)
        if dataset.is_paired:
            # For paired data, collate function extracts the text part (b[1])
            def language_collate_fn(batch):
                return [b[1] for b in batch]

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=language_collate_fn,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
            )
        else:
            # For non-paired data (class labels), create a ClassDataset
            def language_collate_fn(batch):
                return [
                    b for b in batch
                ]  # Batch items are already lists of strings

            class_dataset = ClassDataset(
                classes=dataset.classes,
                prompts=dataset.prompts,
            )

            dataloader = DataLoader(
                class_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=language_collate_fn,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
            )

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Move model to device and set to evaluation mode
        model = model.to(device=device).eval()

        # Load from checkpoint if provided
        if checkpoint_path is not None and checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            self.embedding_list = torch.load(checkpoint_path)
        else:
            self.embedding_list = []

        # Process batches
        for dataloader_index, batch in enumerate(tqdm(dataloader)):
            if dataloader_index * batch_size < len(
                self.embedding_list
            ):  # Skip processed if resuming
                continue

            # batch: list (batch_size) of lists (num_prompts) of strings
            # Initialize list to store embeddings for each item in the batch
            embeddings = [[] for _ in range(len(batch))]
            # Iterate through prompts (or single text if no prompts)
            for i in range(max(len(b) for b in batch)):
                embedding = model(
                    [b[i] for b in batch if i < len(b)],
                )
                embedding = embedding.cpu().to(
                    dtype=dtype
                )  # Move to CPU before storing
                embedding_index = 0
                for batch_index, b in enumerate(batch):
                    if i < len(b):
                        embeddings[batch_index].append(
                            embedding[embedding_index]
                        )
                        embedding_index += 1
            self.embedding_list.extend(embeddings)

        # Pad embeddings to the same length (max number of prompts) and stack
        max_len = max(len(b) for b in self.embedding_list)
        all_embeddings = torch.full(
            (
                len(self.embedding_list),
                max_len,
                *self.embedding_list[0][0].shape,
            ),
            fill_value=torch.nan,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        for i, embeddings in enumerate(self.embedding_list):
            for j, embedding in enumerate(embeddings):
                all_embeddings[i, j] = embedding

        # Save concatenated embeddings
        print(f"Saving embeddings to {path}")
        os.makedirs(path.parent, exist_ok=True)
        torch.save(all_embeddings, path)

        # Clean up
        del embeddings, all_embeddings, batch, model, dataloader
        self.embedding_list = []
        clear_memory()
        return

    def checkpoint(self, *args, **kwargs):
        """
        Saves the current list of computed language embeddings as an intermediate checkpoint.
        This method is typically called by a job scheduler like submitit.

        Args:
            *args: Positional arguments, path is expected as the first argument.
            **kwargs: Keyword arguments, path can also be passed as a keyword.

        Returns:
            The result of the superclass's checkpoint method.
        """
        # Determine the path for the checkpoint file
        path = kwargs.get("path", args[0])
        if isinstance(path, str):
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        checkpoint_path = path.parent / f"intermediate_{path.name}"
        torch.save(self.embedding_list, checkpoint_path)
        print(f"Saving intermediate checkpoint to {checkpoint_path}")

        # Pass the checkpoint_path to the next invocation of __call__
        kwargs["checkpoint_path"] = checkpoint_path
        return super().checkpoint(*args, **kwargs)


class LabelComuputer(Checkpointable):
    """
    Computes and saves labels for a given dataset.
    For paired datasets, labels are sequential indices.
    For other datasets, labels are extracted from the dataset items.
    """

    @torch.inference_mode()
    def __call__(
        self,
        path: Path,
        dataset: Dataset | None,
    ):
        """
        Computes and saves dataset labels.

        Args:
            path: Path to save the computed labels.
            dataset: The Dataset instance to process.
        """
        print(f"Computing labels for {dataset}")
        if dataset.is_paired:
            # For paired datasets, labels are just indices 0 to N-1
            labels = torch.arange(len(dataset), dtype=torch.long)
        else:
            # For non-paired (e.g., classification) datasets, extract actual labels
            labels = [dataset[i][1] for i in trange(len(dataset))]
            labels = torch.tensor(labels, dtype=torch.long)

        # Save labels
        print(f"Saving labels to {path}")
        os.makedirs(path.parent, exist_ok=True)
        torch.save(labels, path)

        # Clean up
        del labels
        clear_memory()
        return
