"""
This script performs shuffle alignment experiments to measure the similarity
between vision and language embeddings under varying degrees of permutation
of the vision embeddings.


It involves:
1. Extracting embeddings for various vision and language model combinations on specified datasets.
2. Computing alignment scores using different distortion metrics.
3. Visualizing the results to analyze the impact of shuffling on alignment scores (Fig. 2, 7).
4. Evaluate random initialized vision models (Fig. 8) and the empirical Lipschitz
   constants of the random models (Fig. 9).
"""

from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from submitit.helpers import Checkpointable
from torch.nn.functional import normalize
from torch.utils.data import Dataset
from tqdm import tqdm

from itsamatch.datasets import (
    CIFAR100,
    CocoCaptions,
    ImageNet100,
)
from itsamatch.distortion_metrics import (
    CKA,
    DistortionMetric,
    GromovWasserstein,
    MutualkNN,
)
from itsamatch.experiments.utils import (
    cluster,
    cococaptions_json,
    cococaptions_root,
    data_root,
    imagenet_root,
    markers,
    markersize,
    palette,
    path_to_embeddings,
    path_to_logs,
    path_to_processed_results,
    path_to_raw_results,
    seed,
)
from itsamatch.extract_embeddings import (
    get_vision_embedding_path,
    load_and_preprocess_embeddings,
    load_embeddings,
    submit_missing_embeddings,
)
from itsamatch.models.language import (
    LanguageModel,
    SentenceT,
)
from itsamatch.models.vision import (
    CLIP,
    DeiT,
    DINOv2,
    PixelValues,
    Random,
    RandomViT,
    VisionModel,
)
from itsamatch.utils import (
    get_seeds,
    seed_everything,
    sweep_or_load_jobs,
)


class ShuffleAlignmentJob(Checkpointable):
    """
    A checkpointable job for performing shuffle alignment experiments.

    This job computes alignment scores between vision and language embeddings
    while progressively shuffling the vision embeddings.
    """

    @torch.inference_mode()
    def __call__(
        self,
        vision_model: VisionModel,
        language_model: LanguageModel,
        dataset: Dataset,
        distortion_metric: DistortionMetric,
        path_to_embeddings: Path,
        num_samples: int = 21,
        num_seeds: int = 20,
        seed: int = 42,
        normalize_before_averaging: bool = False,
    ):
        """
        Executes the shuffle alignment experiment.

        Args:
            vision_model: The vision model to use.
            language_model: The language model to use.
            dataset: The dataset to use.
            distortion_metric: The distortion metric for alignment.
            path_to_embeddings: Path to pre-computed embeddings
            num_samples: The number of shuffling levels to test.
            num_seeds: The number of random seeds for permutation.
            seed: The global random seed.
            normalize_before_averaging: Whether to normalize embeddings before averaging.

        Returns:
            A pandas DataFrame containing the results of the experiment.
        """
        seeds = get_seeds(seed, num_seeds)

        # create shuffling levels: 0, ..., 1 of the embeddings are shuffled
        shuffling_levels = torch.linspace(0, 1, num_samples).tolist()

        # load embeddings
        vision_embeddings, language_embeddings, labels = (
            load_and_preprocess_embeddings(
                vision_model=vision_model,
                language_model=language_model,
                dataset=dataset,
                path_to_embeddings=path_to_embeddings,
                normalize_before_averaging=normalize_before_averaging,
            )
        )

        # average vision embeddings per class
        vision_embeddings = torch.stack(
            [
                vision_embeddings[labels == i].mean(dim=0)
                for i in labels.unique(sorted=True)
            ]
        )

        # normalize embeddings
        vision_embeddings = normalize(vision_embeddings, dim=-1)

        # number of samples and kernel matrices
        num_vision_samples = vision_embeddings.shape[0]

        # compute kernel matrices
        vision_kernel_matrix = distortion_metric.kernel_source(
            vision_embeddings
        )
        language_kernel_matrix = distortion_metric.kernel_target(
            language_embeddings
        )

        results = []

        # iterate over shuffling levels and random seeds
        for shuffling_level in tqdm(shuffling_levels):
            for local_seed in seeds:
                seed_everything(local_seed)

                # permute a fraction of indices
                number_permuted = int(shuffling_level * num_vision_samples)
                permutation_partial = torch.randperm(num_vision_samples)[
                    :number_permuted
                ]
                permutation = torch.arange(num_vision_samples)
                permutation[permutation_partial.sort().values] = (
                    permutation_partial
                )

                # apply permutation to vision kernel matrix
                vision_kernel_matrix_permuted = vision_kernel_matrix[
                    permutation
                ][:, permutation]

                # compute alignment score
                alignment_score = (
                    distortion_metric.loss(
                        vision_kernel_matrix_permuted, language_kernel_matrix
                    )
                    .sum()
                    .item()
                )

                # collect results
                results.append(
                    {
                        "Amount of shuffling": shuffling_level,
                        "local_seed": local_seed,
                        "alignment_score": alignment_score,
                    }
                )

        # build DataFrame and save
        df = pd.DataFrame(results)
        df["global_seed"] = seed
        df["num_samples"] = num_vision_samples
        df["num_seeds"] = num_seeds
        df["vision_model_class"] = vision_model.__class__.__name__
        df["vision_model_name"] = vision_model.name
        df["language_model_class"] = language_model.__class__.__name__
        df["language_model_name"] = language_model.name
        df["dataset"] = str(dataset)
        df["distortion_metric"] = str(distortion_metric)

        return df


def plot_shuffle_alignment(
    df: pd.DataFrame,
    hue_order: list[str] = None,
    markers: dict[str, str] = None,
    palette: dict[str, str] = None,
):
    """
    Plots the shuffle alignment results.

    Args:
        df: DataFrame containing the shuffle alignment results.
        hue_order: Order for the hue categories (vision model classes).
        markers: Markers for different vision model classes.
        palette: Color palette for different vision model classes.

    Returns:
        A tuple containing the matplotlib Figure and Axes objects.
    """
    fig, ax = plt.subplots(figsize=(10, 6.18))
    sns.lineplot(
        data=df,
        x="Amount of shuffling",
        y="alignment_score",
        hue="vision_model_class",
        hue_order=hue_order,
        errorbar="sd",
        style="vision_model_class",
        markers=markers,
        dashes=False,
        markersize=markersize,
        palette=palette,
        ax=ax,
    )
    return fig, ax


if __name__ == "__main__":
    # --- Experiment Configuration ---
    experiment_name = "shuffle_alignment"

    num_jobs = 30  # Number of parallel jobs for computation
    num_samples = 21  # Number of shuffling levels
    num_seeds = (
        100  # Number of random seeds for permutation per shuffling level
    )

    random_model_num_seeds = (
        20  # Number of seeds for random model initializations
    )

    recompute = False  # Flag to recompute results even if they exist

    # Generate seeds for random model initializations
    model_seeds = torch.randint(
        0, 2**32 - 1, (random_model_num_seeds,), dtype=torch.long
    ).tolist()

    # Define vision models to be tested
    vision_models = [
        DINOv2(name="ViT-G/14"),
        CLIP(name="ViT-L/14@336"),
        DeiT(name="DeiT-B/16d@384"),
        PixelValues(),
    ]
    # Add randomly initialized vision models
    for model_seed in model_seeds:
        vision_models.extend(
            [
                RandomViT(name="ViT-H/14", seed=model_seed),
                Random(dim=1280, seed=model_seed),
            ]
        )

    # Define language models to be tested
    language_models = [
        SentenceT(name="all-mpnet-base-v2"),
    ]

    # Initialize COCO Captions dataset
    cococaptions = CocoCaptions(
        root=cococaptions_root,
        annFile=cococaptions_json,
    )

    # Define datasets to be used
    datasets = [
        cococaptions,
        ImageNet100(
            root=data_root,
            imagenet_root=imagenet_root,
            split="val",
        ),
        CIFAR100(
            root=data_root,
            train=False,
            download=True,
        ),
    ]
    # Initialize Gromov-Wasserstein distortion metric
    gw = GromovWasserstein()
    # Define distortion metrics to be used
    distortion_metrics = [
        gw,
        MutualkNN(k=10),
        CKA(),
    ]

    # --- Embedding Extraction ---
    # Submit jobs to compute missing embeddings if they don't exist
    submit_missing_embeddings(
        path_to_embeddings,
        path_to_logs,
        vision_models,
        language_models,
        datasets,
        num_jobs=num_jobs,
        runtime_per_function=1000,
        cluster=cluster,
        name="embed",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=4,
        gpus_per_node=1,
        mem_gb=256,
    )

    # --- Alignment ---
    # Define a condition for running shuffle alignment jobs
    def shuffle_alignment_condition(
        dataset: Dataset,
        distortion_metric: DistortionMetric,
        **kwargs,
    ):
        """Condition to run shuffle alignment. Only GW works with non-paired datasets."""
        return dataset.is_paired or isinstance(
            distortion_metric, GromovWasserstein
        )

    # Run or load shuffle alignment jobs
    df = sweep_or_load_jobs(
        function=ShuffleAlignmentJob(),
        job_kwargs={
            "vision_model": vision_models,
            "language_model": language_models,
            "dataset": datasets,
            "distortion_metric": distortion_metrics,
            "path_to_embeddings": [path_to_embeddings],
            "num_samples": [num_samples],
            "num_seeds": [num_seeds],
            "seed": [seed],
        },
        num_jobs=num_jobs,
        save_path=path_to_raw_results / f"{experiment_name}.csv",
        path_to_logs=path_to_logs,
        runtime_per_function=60,  # minutes
        cluster=cluster,
        recompute=recompute,
        condition=shuffle_alignment_condition,
        shuffle=True,  # Shuffle job order for better load balancing
        submitit_kwargs={
            "name": "alignment",
            "nodes": 1,
            "tasks_per_node": 1,
            "cpus_per_task": 4,
            "gpus_per_node": 0,  # CPU-only jobs
            "mem_gb": 128,
        },
    )

    # --- Result Processing and Saving ---
    # Define path for processed results
    path_to_these_results = path_to_processed_results / experiment_name
    path_to_these_results.mkdir(parents=True, exist_ok=True)

    # alignment score vs. amount of shuffling (Fig. 2, 7, 8 left)
    # Define a subset of models for plotting
    model_subset = [
        "CLIP",
        "DeiT",
        "DINOv2",
        "Random",
    ]

    # Iterate over distortion metrics and datasets to generate plots
    for distortion_metric in distortion_metrics:
        for dataset in datasets:
            # Filter DataFrame for current metric and dataset
            df_filtered = df[
                (df["dataset"] == str(dataset))
                & (df["distortion_metric"] == str(distortion_metric))
                & (df["vision_model_class"].isin(model_subset))
            ].copy()
            if df_filtered.empty:
                continue  # Skip if no data for this combination

            # Adjust alignment score based on metric type
            if isinstance(distortion_metric, MutualkNN) or isinstance(
                distortion_metric, CKA
            ):
                # change from negative inner product (distance) to inner product (similarity)
                df_filtered["alignment_score"] *= -1
            else:
                # scale the alignment score between 0 and 1 for each model
                df_filtered["alignment_score"] = df_filtered.groupby(
                    "vision_model_class"
                )["alignment_score"].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min())
                )

            # Plot shuffle alignment
            fig, ax = plot_shuffle_alignment(
                df=df_filtered,
                hue_order=model_subset,
                markers=markers,
                palette=palette,
            )
            ax.set_ylabel(f"Similarity ({distortion_metric})")
            fig.savefig(
                path_to_these_results / f"{dataset}_{distortion_metric}.png",
                bbox_inches="tight",
            )
            plt.close(fig)

    # implement zoomed in version (Fig. 8 right)
    # Filter for specific dataset and metric for zoomed plot
    distortion_metric = MutualkNN(k=10)
    df_filtered = df[
        (df["dataset"] == "CocoCaptions")
        & (df["distortion_metric"] == str(distortion_metric))
    ].copy()
    df_filtered["alignment_score"] *= -1  # Convert to similarity
    # Plot zoomed shuffle alignment
    fig, ax = plot_shuffle_alignment(
        df=df_filtered,
        hue_order=[
            "CLIP",
            "DeiT",
            "DINOv2",
            "Random ViT",
            "Pixel values",
            "Random",
        ],
        markers=markers,
        palette=palette,
    )
    ax.set_ylim(0, 1e-2)  # Set y-axis limit for zoom
    ax.set_ylabel(
        f"Similarity ({distortion_metric})"
    )  # Use the last distortion_metric in loop, should be Mutual k-NN
    fig.savefig(
        path_to_processed_results
        / "shuffle_alignment"
        / "Zoomed_CocoCaptions_Mutual k-NN.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # compute the empirical Lipschitz constants (Fig. 9)
    # Load pixel value embeddings
    pixel_values = load_embeddings(
        get_vision_embedding_path(
            root=path_to_embeddings,
            model=PixelValues(),
            dataset=cococaptions,
        )
    )

    # Load embeddings for multiple random ViT initializations
    random_vits = torch.stack(
        [
            load_embeddings(
                get_vision_embedding_path(
                    root=path_to_embeddings,
                    model=RandomViT(name="ViT-H/14", seed=model_seed),
                    dataset=cococaptions,
                )
            )
            for model_seed in model_seeds
        ]
    )

    # Normalize embeddings
    pixel_values = normalize(pixel_values, dim=-1)
    random_vits = normalize(random_vits, dim=-1)

    # Compute kernel matrices (distances)
    pixel_dist = gw.kernel_source(
        pixel_values
    )  # Pairwise distances for pixel values
    random_dist = gw.kernel_target(
        random_vits
    )  # Pairwise distances for random ViTs (averaged over seeds)

    # Compute empirical Lipschitz constants
    # Ratio of distances in output space (random_vits) to input space (pixel_values)
    lipschitz_constants = (
        random_dist.double().cpu() / pixel_dist[None].double().cpu()
    )  # Add batch dim to pixel_dist for broadcasting
    finite_lipschitz_constants = lipschitz_constants[
        lipschitz_constants.isfinite()
    ]  # Filter out non-finite values

    # Create histogram of Lipschitz constants
    # only <0.01% of the values are larger than 2 so we can safely ignore them
    hist = torch.histogram(
        finite_lipschitz_constants[
            finite_lipschitz_constants < 2
        ],  # Filter outliers
        bins=50,
        density=True,
    )
    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6.18))
    sns.barplot(
        x=(hist.bin_edges[1:] + hist.bin_edges[:-1]) / 2,  # Bin centers
        y=hist.hist,  # Bin heights (density)
        native_scale=True,
        ax=ax,
        color="#0065BD",
    )
    ax.set_xlim(0, 2)
    ax.set_xlabel(
        r"Empirical Lipschitz constant $\frac{\|f(x) - f(y)\|_2}{\|x - y\|_2}$"
    )
    ax.set_ylabel("Density")
    fig.savefig(
        path_to_processed_results
        / "shuffle_alignment"
        / "Lipschitz_constants.png",
        bbox_inches="tight",
    )
    plt.close(fig)
