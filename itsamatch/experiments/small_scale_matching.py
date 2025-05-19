"""
This script conducts small-scale matching experiments (Sec. 5.1).

It involves:
1. Extracting embeddings for various vision and language model combinations on specified datasets.
2. Computing matching accuracies using different distortion metrics.
3. Generating and saving plots:
    - Matching accuracy plots (Fig. 4).
    - Distortion metric comparison table (Tab. 3).
    - Matching accuracy heatmaps (Fig. 12, 13).
    - Model size vs. matching accuracy/similarity plots (Fig. 11).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from matplotlib import pyplot as plt
from submitit.helpers import Checkpointable
from torch.nn.functional import normalize
from torch.utils.data import Dataset

from itsamatch.datasets import (
    CIFAR10,
    CINIC10,
)
from itsamatch.distortion_metrics import (
    CKA,
    DistortionMetric,
    GromovWasserstein,
    MutualkNN,
)
from itsamatch.experiments.larger_scale_matching import UnsupervisedMatching
from itsamatch.experiments.utils import (
    cluster,
    data_root,
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
    load_and_preprocess_embeddings,
    submit_missing_embeddings,
)
from itsamatch.models.language import (
    CLIP as CLIPLanguage,
)
from itsamatch.models.language import (
    LanguageModel,
    SentenceT,
)
from itsamatch.models.vision import (
    CLIP,
    DINO,
    ConvNeXt,
    DeiT,
    DINOv2,
    VisionModel,
)
from itsamatch.solvers import BruteForce
from itsamatch.utils import (
    sweep_or_load_jobs,
)


def plot_accuracy(
    df: pd.DataFrame,
    vision_models: list,
    language_model_subset: list,
    datasets: list,
    distortion_metric: str,
    path_to_these_results: Path,
    horizontal: bool = False,
):
    """Plots matching accuracy for different vision models.

    Generates a point plot comparing matching accuracy across vision models
    for specified datasets and language models, using a given distortion metric.
    The plot can be oriented horizontally or vertically.

    Args:
        df: DataFrame containing the matching accuracy results.
        vision_models: List of vision model instances.
        language_model_subset: List of language model names to include.
        datasets: List of dataset instances.
        distortion_metric: Name of the distortion metric used.
        path_to_these_results: Path to save the generated plot.
        horizontal: If True, plot is horizontal; otherwise, vertical.
    """
    if horizontal:
        fig, axs = plt.subplots(2, figsize=(20, 12.36))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12.36, 20))
    df["vision_model"] = (
        df["vision_model_class"] + "_" + df["vision_model_name"]
    )
    num_vision_models = len(vision_models)
    for ax, dataset, language_model_name in zip(
        axs, datasets, language_model_subset
    ):
        df_filtered = df[
            (df["dataset"] == str(dataset))
            & (df["language_model_name"] == language_model_name)
            & (df["distortion_metric"] == distortion_metric)
        ]
        for vision_model in vision_models:
            vision_class = vision_model.__class__.__name__
            vision_name = vision_model.name
            df_vision_class = df_filtered[
                (df_filtered["vision_model_class"] == vision_class)
                & (df_filtered["vision_model_name"] == vision_name)
            ]

            if horizontal:
                x = "vision_model"
                y = "Matching accuracy"
                capsize = 0.2
            else:
                x = "Matching accuracy"
                y = "vision_model"
                capsize = 0.1

            sns.pointplot(
                data=df_vision_class,
                x=x,
                y=y,
                linestyle="none",
                errorbar="sd",
                capsize=capsize,
                err_kws={"color": "black", "linewidth": 2},
                markers=markers[vision_class],
                markersize=markersize,
                markeredgewidth=3,
                ax=ax,
                color=palette[vision_class],
                label=vision_class,
            )
        ax.set_title(f"{dataset}\n{language_model_name}")

        if horizontal:
            ax.set_ylim(0, 1)
            ax.set_xlim(-1, num_vision_models + 1)
            ax.plot(
                [-1, num_vision_models + 1],
                [0.1, 0.1],
                c="black",
                ls="dashed",
                label="Random",
            )
            ax.set(xlabel=None)
            ax.set_ylabel("Matching accuracy")
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, num_vision_models)
            ax.plot(
                [0.1, 0.1],
                [-1, num_vision_models + 1],
                c="black",
                ls="dashed",
                label="Random",
            )
            ax.set(ylabel=None)
            ax.set_xlabel("Matching accuracy")
    handles0, labels0 = axs[0].get_legend_handles_labels()
    handles1, labels1 = axs[0].get_legend_handles_labels()
    handles, labels = handles0 + handles1, labels0 + labels1
    labels_unique, index = np.unique(labels, return_index=True)
    labels_unique = labels_unique.tolist()
    handles_unique = [handles[i] for i in index]

    if horizontal:
        axs[0].set_xticks([])
        axs[1].set_xticks(
            list(range(len(vision_models))),
            [vision_model.name for vision_model in vision_models],
            rotation=45,
            ha="right",
            fontsize=22,
        )
        axs[0].legend(
            handles=handles_unique,
            labels=labels_unique,
            loc="upper left",
            fontsize=18,
        )
        axs[1].legend_ = None
    else:
        axs[1].set_yticks([])
        axs[0].set_yticks(
            list(range(len(vision_models))),
            [vision_model.name for vision_model in vision_models],
            ha="right",
            fontsize=22,
        )
        axs[1].legend(
            handles=handles_unique,
            labels=labels_unique,
            loc="upper right",
            fontsize=18,
        )
        axs[0].legend_ = None
        for ax in axs:
            ax.yaxis.set_inverted(True)

    path_to_these_results.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_to_these_results / "fig_4.png", bbox_inches="tight")
    plt.close(fig)


class AlignmentScore(Checkpointable):
    """Computes the alignment score between vision and language embeddings."""

    @torch.inference_mode()
    def __call__(
        self,
        vision_model: VisionModel,
        language_model: LanguageModel,
        dataset: Dataset,
        distortion_metric: DistortionMetric,
        normalize_before_averaging: bool = False,
    ):
        """Calculates the alignment score.

        Args:
            vision_model: The vision model instance.
            language_model: The language model instance.
            dataset: The dataset instance.
            distortion_metric: The distortion metric instance to use.
            normalize_before_averaging: Whether to normalize embeddings before averaging.

        Returns:
            A DataFrame containing the alignment score and model/dataset details.
        """
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

        # average vision embeddings
        vision_embeddings = torch.stack(
            [
                vision_embeddings[labels == i].mean(dim=0)
                for i in labels.unique(sorted=True)
            ]
        )

        # normalize embeddings
        vision_embeddings = normalize(vision_embeddings, dim=-1)

        # compute kernel matrices
        vision_kernel_matrix = distortion_metric.kernel_source(
            vision_embeddings
        )
        language_kernel_matrix = distortion_metric.kernel_target(
            language_embeddings
        )

        alignment_score = (
            distortion_metric.loss(vision_kernel_matrix, language_kernel_matrix)
            .sum()
            .item()
        )

        # build DataFrame and save
        df = pd.DataFrame(
            {
                "vision_model_class": vision_model.__class__.__name__,
                "vision_model_name": vision_model.name,
                "language_model_class": language_model.__class__.__name__,
                "language_model_name": language_model.name,
                "dataset": str(dataset),
                "distortion_metric": str(distortion_metric),
                "alignment_score": alignment_score,
            },
            index=[0],
        )
        return df


def plot_accuracy_heatmap(
    df: pd.DataFrame,
    vision_models: list,
    language_models: list,
    distortion_metric: str,
    dataset: str,
    path_to_these_results: Path,
    file_name: str,
):
    """Plots a heatmap of matching accuracies.

    Generates a heatmap showing matching accuracy for combinations of
    vision and language models on a specific dataset and distortion metric.

    Args:
        df: DataFrame containing the matching accuracy results.
        vision_models: List of vision model instances.
        language_models: List of language model instances.
        distortion_metric: Name of the distortion metric used.
        dataset: Name of the dataset.
        path_to_these_results: Path to save the generated plot.
        file_name: Name of the file to save the plot as.
    """

    def get_vision_model_name(class_name, model_name):
        if class_name in ["ConvNeXt", "DeiT"]:
            return model_name
        else:
            return f"{class_name} {model_name}"

    def get_language_model_name(class_name, model_name):
        if class_name in ["SentenceT"]:
            return model_name
        else:
            return f"{class_name} {model_name}"

    df["Vision model"] = df.apply(
        lambda row: get_vision_model_name(
            row["vision_model_class"], row["vision_model_name"]
        ),
        axis=1,
    )
    df["Language model"] = df.apply(
        lambda row: get_language_model_name(
            row["language_model_class"], row["language_model_name"]
        ),
        axis=1,
    )

    vision_order = [
        get_vision_model_name(
            vision_model.__class__.__name__, vision_model.name
        )
        for vision_model in vision_models
    ]
    language_order = [
        get_language_model_name(
            language_model.__class__.__name__, language_model.name
        )
        for language_model in language_models
    ]

    df_filtered = df[
        (df["dataset"] == dataset)
        & (df["distortion_metric"] == distortion_metric)
    ]

    pivot = df_filtered.pivot_table(
        values="Matching accuracy",
        index=["Vision model"],
        columns=["Language model"],
        aggfunc="mean",
    )
    pivot = pivot.loc[vision_order, language_order]

    fig, ax = plt.subplots(figsize=(24.72, 40))
    sns.heatmap(
        pivot,
        square=True,
        cbar_kws={
            "orientation": "horizontal",
            "location": "bottom",
            "anchor": (0.5, 2.0),
            "label": "Matching accuracy",
        },
        cmap="viridis",
        ax=ax,
    )
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xticklabels(language_order, rotation=30, ha="left")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    fig.savefig(path_to_these_results / file_name, bbox_inches="tight")
    plt.close(fig)


def plot_model_size(
    df: pd.DataFrame,
    language_models_subset: list[LanguageModel],
    vision_models: list[VisionModel],
    datasets: list[Dataset],
    distortion_metric: DistortionMetric,
    path_to_these_results: Path,
    path_to_raw_results: Path,
    path_to_logs: Path,
    experiment_name: str,
    cluster: str,
):
    """Plots matching accuracy and similarity against vision model size.

    Generates scatter plots with linear model fits, showing the relationship
    between vision model size and (1) matching accuracy, and (2) alignment similarity.
    Plots are faceted by dataset, language model, and metric.

    Args:
        df: DataFrame containing the matching accuracy results.
        language_models_subset: List of language model instances for this plot.
        vision_models: List of all vision model instances.
        datasets: List of dataset instances.
        distortion_metric: The distortion metric instance for alignment score calculation.
        path_to_these_results: Path to save the generated plot.
        path_to_raw_results: Path to raw results (for model sizes and alignment scores).
        path_to_logs: Path to logs for job submission.
        experiment_name: Name of the experiment.
        cluster: Cluster environment for job submission.
    """
    language_model_names = [
        language_model.name for language_model in language_models_subset
    ]

    def get_model_size(model):
        return sum(p.numel() for p in model.parameters())

    vision_model_size_file = path_to_raw_results / "vision_model_sizes.yml"
    if vision_model_size_file.exists():
        with open(vision_model_size_file, "r") as f:
            vision_model_sizes = yaml.safe_load(f)
    else:
        vision_model_sizes = {}
        for vision_model in vision_models:
            vision_model = vision_model.to(torch.device("cpu"))
            key = f"{vision_model.__class__.__name__}_{vision_model.name}"
            vision_model_sizes[key] = get_model_size(vision_model.model)
            del vision_model.model
        with open(vision_model_size_file, "w") as f:
            yaml.dump(vision_model_sizes, f)

    df_filtered = df[
        (df["distortion_metric"] == "GW Distance")
        & df["language_model_name"].isin(language_model_names)
    ]

    # average over seeds
    df_filtered = df_filtered.groupby(
        [
            "vision_model_class",
            "vision_model_name",
            "language_model_class",
            "language_model_name",
            "dataset",
            "distortion_metric",
        ],
        as_index=False,
    ).mean(numeric_only=True)

    df_alignment = sweep_or_load_jobs(
        function=AlignmentScore(),
        job_kwargs={
            "vision_model": vision_models,
            "language_model": language_models_subset,
            "dataset": datasets,
            "distortion_metric": [distortion_metric],
        },
        num_jobs=1,
        save_path=path_to_raw_results / f"{experiment_name}_alignment.csv",
        path_to_logs=path_to_logs,
        runtime_per_function=10,
        cluster=cluster,
        recompute=False,
        condition=None,
        shuffle=True,
        submitit_kwargs={
            "nodes": 1,
            "tasks_per_node": 1,
            "cpus_per_task": 4,
            "gpus_per_node": 0,
            "mem_gb": 64,
        },
    )

    # merge df with df_alignment
    df_filtered["metric"] = "Matching accuracy"
    df_alignment["metric"] = "Similarity (Mutual k-NN)"
    df_filtered["value"] = df_filtered["Matching accuracy"]
    df_alignment["value"] = df_alignment["alignment_score"]
    df_filtered = pd.concat(
        [df_filtered, df_alignment], ignore_index=True
    ).reset_index(drop=True)
    df_filtered["Vision model size"] = (
        df_filtered["vision_model_class"]
        + "_"
        + df_filtered["vision_model_name"]
    ).map(vision_model_sizes)

    hue_order = ["CLIP", "ConvNeXt", "DeiT", "DINO", "DINOv2"]
    marker_list = [markers[hue] for hue in hue_order]
    df_filtered["dataset + language_model_name"] = (
        df_filtered["dataset"] + df_filtered["language_model_name"]
    )
    col_order = [
        str(dataset) + language_model_name
        for dataset in datasets
        for language_model_name in language_model_names
    ]

    g = sns.lmplot(
        data=df_filtered,
        x="Vision model size",
        y="value",
        col="dataset + language_model_name",
        col_order=col_order,
        row="metric",
        hue="vision_model_class",
        ci=None,
        logx=True,
        palette=palette,
        hue_order=hue_order,
        markers=marker_list,
        aspect=1.618033988,
        facet_kws=dict(
            sharex=True,
            sharey=False,
            legend_out=False,
        ),
        scatter_kws=dict(alpha=0.5, s=80),
    )

    for ax, name in zip(g.axes[0], language_model_names * len(datasets)):
        ax.set_title(name)

    for ax in g.axes[1]:
        ax.set_title(None)

    fig = g.figure
    inv = fig.transFigure.inverted()
    for dataset_index, dataset in enumerate(datasets):
        left_ax = g.axes[0][2 * dataset_index]
        right_ax = g.axes[0][2 * dataset_index + 1]
        x_pos = (
            left_ax.get_window_extent().x1 + right_ax.get_window_extent().x0
        ) / 2
        x_pos = inv.transform((x_pos, 1))[0]
        fig.text(
            x_pos, 1.0, str(dataset), va="center", ha="center", size="large"
        )

    for row in g.axes:
        for ax in row[1:]:
            ax.sharey(row[0])
            ax.tick_params(labelleft=False)

    labels, handles = zip(*g._legend_data.items())
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(hue_order),
    )
    g.axes[0, 0].get_legend().remove()

    g.axes[0, 0].set_ylabel("Matching accuracy")
    g.axes[1, 0].set_ylabel("Similarity (Mutual k-NN)")

    plt.semilogx()

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(path_to_these_results / "fig_11.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # --- Experiment Configuration ---
    experiment_name = "small_scale_matching"

    num_jobs = 30

    vision_models = [
        CLIP(name="RN50"),
        CLIP(name="RN101"),
        CLIP(name="RN50x4"),
        CLIP(name="RN50x16"),
        CLIP(name="RN50x64"),
        CLIP(name="ViT-B/32"),
        CLIP(name="ViT-B/16"),
        CLIP(name="ViT-L/14"),
        CLIP(name="ViT-L/14@336"),
        ConvNeXt(name="CN-B-1"),
        ConvNeXt(name="CN-B-22"),
        ConvNeXt(name="CN-L-1"),
        ConvNeXt(name="CN-L-22"),
        ConvNeXt(name="CN-L-22ft@384"),
        ConvNeXt(name="CN-XL-22ft@384"),
        DeiT(name="DeiT-T/16"),
        DeiT(name="DeiT-T/16d"),
        DeiT(name="DeiT-S/16"),
        DeiT(name="DeiT-S/16d"),
        DeiT(name="DeiT-B/16"),
        DeiT(name="DeiT-B/16@384"),
        DeiT(name="DeiT-B/16d"),
        DeiT(name="DeiT-B/16d@384"),
        DINO(name="RN50"),
        DINO(name="ViT-S/16"),
        DINO(name="ViT-S/8"),
        DINO(name="ViT-B/16"),
        DINO(name="ViT-B/8"),
        DINOv2(name="ViT-S/14"),
        DINOv2(name="ViT-B/14"),
        DINOv2(name="ViT-L/14"),
        DINOv2(name="ViT-G/14"),
    ]

    language_models = [
        SentenceT(name="all-MiniLM-L6-v1"),
        SentenceT(name="all-MiniLM-L6-v2"),
        SentenceT(name="all-MiniLM-L12-v1"),
        SentenceT(name="all-MiniLM-L12-v2"),
        SentenceT(name="msmarco-distilbert-dot-v5"),
        SentenceT(name="average_word_embeddings_komninos"),
        SentenceT(name="all-distilroberta-v1"),
        SentenceT(name="msmarco-bert-base-dot-v5"),
        SentenceT(name="all-mpnet-base-v2"),
        SentenceT(name="all-mpnet-base-v1"),
        SentenceT(name="gtr-t5-base"),
        SentenceT(name="sentence-t5-base"),
        SentenceT(name="average_word_embeddings_glove.6B.300d"),
        SentenceT(name="gtr-t5-large"),
        SentenceT(name="sentence-t5-large"),
        SentenceT(name="All-Roberta-large-v1"),
        SentenceT(name="gtr-t5-xl"),
        SentenceT(name="sentence-t5-xl"),
        CLIPLanguage(name="RN50"),
        CLIPLanguage(name="RN101"),
        CLIPLanguage(name="RN50x4"),
        CLIPLanguage(name="RN50x16"),
        CLIPLanguage(name="RN50x64"),
        CLIPLanguage(name="ViT-B/32"),
        CLIPLanguage(name="ViT-B/16"),
        CLIPLanguage(name="ViT-L/14"),
        CLIPLanguage(name="ViT-L/14@336px"),
    ]

    datasets = [
        CIFAR10(
            root=data_root,
            train=False,
            download=True,
        ),
        CINIC10(
            root=data_root,
            split="val",
            download=True,
        ),
    ]

    num_seeds = 20

    solver = BruteForce()

    distortion_metrics = [
        GromovWasserstein(),
        CKA(),
    ] + [MutualkNN(k=k) for k in range(1, 9)]

    recompute_matching = False

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

    # --- Matching ---
    df = sweep_or_load_jobs(
        function=UnsupervisedMatching(),
        job_kwargs={
            "vision_model": vision_models,
            "language_model": language_models,
            "dataset": datasets,
            "solver": [solver],
            "distortion_metric": distortion_metrics,
            "path_to_embeddings": [path_to_embeddings],
            "num_seeds": [num_seeds],
            "seed": [seed],
        },
        num_jobs=num_jobs,
        save_path=path_to_raw_results / f"{experiment_name}.csv",
        path_to_logs=path_to_logs,
        runtime_per_function=10,
        cluster=cluster,
        recompute=recompute_matching,
        condition=None,
        shuffle=True,
        submitit_kwargs={
            "name": "matching",
            "nodes": 1,
            "tasks_per_node": 1,
            "cpus_per_task": 4,
            "gpus_per_node": 0,
            "mem_gb": 64,
        },
    )

    # --- Result Processing and Saving ---
    # Define path for processed results
    path_to_these_results = path_to_processed_results / experiment_name
    path_to_these_results.mkdir(parents=True, exist_ok=True)

    # plot and save the results (Fig. 4)
    plot_accuracy(
        df=df,
        vision_models=vision_models,
        language_model_subset=["all-mpnet-base-v2", "All-Roberta-large-v1"],
        datasets=datasets,
        distortion_metric="GW Distance",
        path_to_these_results=path_to_these_results,
        horizontal=False,
    )

    # Compare different distortion metrics (Tab. 3)
    df_filtered = df[
        (df["vision_model_class"] == "DINOv2")
        & (df["vision_model_name"] == "ViT-G/14")
        & (df["language_model_name"] == "all-mpnet-base-v2")
    ]
    mean = (
        (
            df_filtered.pivot_table(
                values="Matching accuracy",
                index=["distortion_metric"],
                columns=["dataset"],
                aggfunc="mean",
            ).round(2)
            * 100
        )
        .astype(int)
        .astype(str)
    )
    std = (
        (
            df_filtered.pivot_table(
                values="Matching accuracy",
                index=["distortion_metric"],
                columns=["dataset"],
                aggfunc="std",
            ).round(2)
            * 100
        )
        .astype(int)
        .astype(str)
    )
    out = (mean + " ± " + std).loc[["Mutual k-NN (k=4)", "CKA", "GW Distance"]]
    out.to_csv(path_to_these_results / "tab_3.csv")

    # Plot all model combination results (Fig. 12, 13)
    plot_accuracy_heatmap(
        df,
        vision_models,
        language_models,
        distortion_metric="GW Distance",
        dataset="CIFAR-10",
        path_to_these_results=path_to_these_results,
        file_name="fig_12.png",
    )

    plot_accuracy_heatmap(
        df,
        vision_models,
        language_models,
        distortion_metric="GW Distance",
        dataset="CINIC-10",
        path_to_these_results=path_to_these_results,
        file_name="fig_13.png",
    )

    # Compare for different model sizes (Fig. 11)
    distortion_metric = MutualkNN(k=4)
    language_models_subset = [
        SentenceT(name="All-Roberta-large-v1"),
        SentenceT(name="all-mpnet-base-v2"),
    ]

    plot_model_size(
        df,
        language_models_subset,
        vision_models,
        datasets,
        distortion_metric,
        path_to_these_results=path_to_these_results,
        path_to_raw_results=path_to_raw_results,
        path_to_logs=path_to_logs,
        experiment_name=experiment_name,
        cluster=cluster,
    )
