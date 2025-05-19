"""
This script conducts the small-scale solver comparison (Sec. 5.3 first paragraph).

It involves:
1. Extracting embeddings for various vision and language model combinations on specified datasets.
2. Computing matching accuracies and costs using different solvers and distortion metrics.
3. Saving the results in a CSV file (Tab. 1, 7).
"""

from pathlib import Path

import pandas as pd

from itsamatch.datasets import (
    CIFAR10,
    CINIC10,
)
from itsamatch.distortion_metrics import (
    CKA,
    GromovWasserstein,
)
from itsamatch.experiments.larger_scale_matching import UnsupervisedMatching
from itsamatch.experiments.utils import (
    cluster,
    data_root,
    path_to_embeddings,
    path_to_logs,
    path_to_processed_results,
    path_to_raw_results,
    seed,
)
from itsamatch.extract_embeddings import submit_missing_embeddings
from itsamatch.models.language import SentenceT
from itsamatch.models.vision import (
    CLIP,
    DeiT,
    DINOv2,
)
from itsamatch.solvers import (
    BruteForce,
    FactorizedHahnGrant,
    Gurobi,
    MPOpt,
    OptimalTransport,
    QAPSolver,
    RandomSolver,
    ScipySolver,
)
from itsamatch.utils import sweep_or_load_jobs


def rename_solvers(df: pd.DataFrame) -> pd.DataFrame:
    """Renames solver names in the DataFrame for consistency with the paper.

    Args:
        df: DataFrame containing experiment results with a 'solver' column.

    Returns:
        DataFrame with renamed solver names.
    """
    # Rename the solver name for the Scipy solver if distortion metric is CKA
    df.loc[
        (df["solver"] == "ScipySolver") & (df["distortion_metric"] == "CKA"),
        "solver",
    ] = "LocalCKA"

    # Rename other Scipy solver to FAQ
    df.loc[
        (df["solver"] == "ScipySolver") & (df["distortion_metric"] != "CKA"),
        "solver",
    ] = "FAQ"

    # Rename the solver name from OptimalTransport to OT
    df.loc[
        (df["solver"] == "OptimalTransport"),
        "solver",
    ] = "OT"

    # Rename the solver name from RandomSolver to Random
    df.loc[
        (df["solver"] == "RandomSolver"),
        "solver",
    ] = "Random"

    # Rename the solver name from FactorizedHahnGrant to Ours
    df.loc[
        (df["solver"] == "FactorizedHahnGrant"),
        "solver",
    ] = "Ours"
    return df


def save_table(
    df: pd.DataFrame,
    save_path: Path,
):
    """Processes the results DataFrame and saves it as a CSV table.

    The processing involves:
    - Renaming solvers.
    - Normalizing costs.
    - Comparing results to a brute-force ground truth to determine global optimality.
    - Aggregating results (mean and std) across different configurations.
    - Formatting and saving the final table.

    Args:
        df: DataFrame containing raw experiment results.
        save_path: Path to save the processed CSV table.
    """
    # Rename solver names for clarity
    df = rename_solvers(df)

    # normalize cost by the square of the problem size
    df["Cost"] = df["GromovWasserstein()"] / df["Problem size"] ** 2

    # compare with brute force solver count as correct if cost is the same or when each element is the same
    gw_str = "GromovWasserstein()"
    eps = 1e-8  # Epsilon for float comparison
    suffix = "_gt"  # Suffix for ground truth columns

    # Convert string representation of predictions to lists if necessary
    if isinstance(df["prediction"].iloc[0], str):
        df["prediction"] = df["prediction"].map(eval)

    # Separate ground truth (BruteForce) results from other solver results
    df_gt = df[df["solver"] == "BruteForce"]
    df_other = df[df["solver"] != "BruteForce"]

    # Define columns for merging ground truth data
    on = [
        "dataset",
        "vision_model_class",
        "vision_model_name",
        "language_model_class",
        "language_model_name",
        "local_seed",
        "global_seed",
        "num_samples",
        "subset",
        "Problem size",
        "timelimit",
    ]
    # Merge other solver results with ground truth results
    df_merged = pd.merge(
        df_other, df_gt, how="left", on=on, suffixes=["", suffix]
    )

    def is_global(row):
        """Checks if a solution is globally optimal compared to the ground truth."""
        return (
            row[gw_str] <= row[f"{gw_str}{suffix}"] + eps
            or row["prediction"] == row["prediction_gt"]
        )

    # Determine global optimality and convert matching accuracy to percentage
    df_merged["Global?"] = df_merged.apply(is_global, axis=1) * 100
    df_merged["Matching accuracy"] = df_merged["Matching accuracy"] * 100

    values = ["Matching accuracy", "Cost", "Global?"]

    # aggregate the results by calculating mean
    mean = (
        df_merged.pivot_table(
            values=values,
            index=["dataset", "vision_model_class", "solver"],
            columns="language_model_name",
            aggfunc="mean",
        )
        .round(2)
        .astype(str)
        .swaplevel(0, 1, axis=1)
    )

    # Aggregate the results by calculating standard deviation
    std = (
        df_merged.pivot_table(
            values=values,
            index=["dataset", "vision_model_class", "solver"],
            columns="language_model_name",
            aggfunc="std",
        )
        .round(2)
        .astype(str)
        .swaplevel(0, 1, axis=1)
    )

    # Define the order of rows and columns for the final table
    dataset_order = [str(dataset) for dataset in datasets]
    vision_model_order = [
        vision_model.__class__.__name__ for vision_model in vision_models
    ]
    solver_order = [
        "Random",
        "LocalCKA",
        "OT",
        "FAQ",
        "MPOpt",
        "Gurobi",
        "Ours",
    ]
    language_model_order = [
        language_model.name for language_model in language_models
    ]
    # Combine mean and std, format, and select data in the desired order
    out = (mean + " ± " + std).loc[
        (dataset_order, vision_model_order, solver_order),
        (language_model_order, values),
    ]
    # Save the formatted table to CSV
    out.to_csv(save_path)


if __name__ == "__main__":
    # --- Experiment Configuration ---
    experiment_name = "solver_comparison_small"

    num_jobs = 30  # Number of parallel jobs for computations

    # Define vision models to be used in the experiment
    vision_models = [
        CLIP(name="ViT-L/14@336"),
        DeiT(name="DeiT-B/16d@384"),
        DINOv2(name="ViT-G/14"),
    ]

    # Define language models to be used in the experiment
    language_models = [
        SentenceT(name="all-mpnet-base-v2"),
        SentenceT(name="All-Roberta-large-v1"),
    ]

    # Define datasets to be used in the experiment
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

    num_seeds = 20  # Number of random seeds for averaging results

    # Define solvers to be compared
    solvers = [
        BruteForce(),
        OptimalTransport(),
        ScipySolver(method="faq", num_repetitions=1),
        ScipySolver(method="faq", num_repetitions=1, transpose_first=True),
        Gurobi(),
        MPOpt(),
        FactorizedHahnGrant(),
        RandomSolver(),
    ]

    recompute = False  # Flag to force recomputation of results

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
    # Define a helper function to determine the distortion metric based on the solver
    def get_distortion_metrics(
        solver: QAPSolver,
        **kwargs,
    ):
        """Dynamically returns the distortion metric based on the solver type."""
        # LocalCKA (ScipySolver with faq, num_repetitions=1, transpose_first=True) uses CKA
        if (
            isinstance(solver, ScipySolver)
            and solver.method == "faq"
            and solver.num_repetitions == 1
            and solver.transpose_first
        ):
            return [CKA()]
        # All other solvers use GromovWasserstein
        else:
            return [GromovWasserstein()]

    # Run the unsupervised matching experiments or load existing results
    df = sweep_or_load_jobs(
        function=UnsupervisedMatching(),
        job_kwargs={
            "vision_model": vision_models,
            "language_model": language_models,
            "dataset": datasets,
            "solver": solvers,
            "distortion_metric": get_distortion_metrics,
            "path_to_embeddings": [path_to_embeddings],
            "num_seeds": [num_seeds],
            "seed": [seed],
            "reference_distortion_metrics": [[GromovWasserstein()]],
        },
        num_jobs=num_jobs,
        save_path=path_to_raw_results / f"{experiment_name}.csv",
        path_to_logs=path_to_logs,
        runtime_per_function=10,
        cluster=cluster,
        recompute=recompute,
        condition=None,
        shuffle=True,
        submitit_kwargs={
            "name": experiment_name,
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

    # Process the DataFrame and save the results table (Tab. 1, 7)
    save_table(
        df=df,
        save_path=path_to_these_results / "tab_1_7.csv",
    )
