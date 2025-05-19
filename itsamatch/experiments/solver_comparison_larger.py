"""
This script conducts the larger-scale solver comparison (Sec. 5.3 second paragraph).

It involves:
1. Extracting embeddings for various vision and language model combinations on specified datasets.
2. Computing matching accuracies and costs using different solvers and problem sizes.
3. Plotting the cost and bound for different problem sizes (Fig. 6).
"""

from pathlib import Path
from typing import Any, Callable
from warnings import warn

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from itsamatch.datasets import CIFAR100
from itsamatch.distortion_metrics import (
    CKA,
    GromovWasserstein,
)
from itsamatch.experiments.larger_scale_matching import (
    UnsupervisedMatching,
    compute_missing_subsets,
    load_subsets_from_kwargs,
)
from itsamatch.experiments.solver_comparison_small import rename_solvers
from itsamatch.experiments.utils import (
    cluster,
    dash,
    data_root,
    gurobi_nodelist,
    markers,
    markersize,
    palette,
    path_to_embeddings,
    path_to_logs,
    path_to_processed_results,
    path_to_raw_results,
    path_to_subsets,
    same_cpu_nodelist,
    seed,
)
from itsamatch.extract_embeddings import submit_missing_embeddings
from itsamatch.models.language import SentenceT
from itsamatch.models.vision import CLIP
from itsamatch.solvers import (
    FactorizedHahnGrant,
    Gurobi,
    MPOpt,
    OptimalTransport,
    QAPSolver,
    RandomSolver,
    ScipySolver,
)
from itsamatch.utils import (
    load_df_from_jobs,
    monitor_jobs,
    sweep,
)


def sweep_or_load_jobs_separate_gurobi(
    function: Callable,
    job_kwargs: dict[str, list | Callable[[Any], list]],
    num_jobs: int,
    save_path: Path,
    path_to_logs: Path,
    runtime_per_function: int,
    cluster: str,
    recompute: bool = False,
    condition: Callable | None = None,
    shuffle: bool = True,
    max_num_timeout: int = 3,
    submitit_kwargs: dict | None = None,
    force: bool = True,
    compute_gurobi_separately: bool = True,
    gurobi_nodelist: str | None = None,
):
    """
    Sweeps a function over a set of job arguments or loads existing results.

    This function handles job submission, monitoring, and result aggregation.
    It has a special mode to run Gurobi jobs on a separate nodelist if specified.

    Args:
        function: The function to be executed for each job.
        job_kwargs: A dictionary where keys are argument names and values are lists
                    of possible values for that argument, or a callable that returns such a list.
                    Gurobi() with GromovWasserstein() is added to the list of solvers.
        num_jobs: The total number of jobs to run.
        save_path: Path to save the aggregated results CSV file.
        path_to_logs: Path to store logs for the jobs.
        runtime_per_function: Estimated runtime per function call in minutes.
        cluster: The cluster environment to run jobs on.
        recompute: If True, recomputes results even if `save_path` exists.
        condition: A callable that takes job parameters and returns True if the job should run.
        shuffle: If True, shuffles the order of jobs before submission.
        max_num_timeout: Maximum number of timeouts allowed before stopping job submission.
        submitit_kwargs: Additional arguments for the submitit executor.
        force: If True, forces loading of results even if some jobs failed.
        compute_gurobi_separately: If True, Gurobi jobs are run on `gurobi_nodelist`.
        gurobi_nodelist: The specific nodelist for Gurobi jobs.

    Returns:
        A pandas DataFrame containing the aggregated results from all jobs.
    """
    # Check if results already exist and load them if recompute is False
    if save_path.exists() and not recompute:
        print(f"Path {save_path} already exists. Loading file...")
        return pd.read_csv(save_path)
    else:
        print(f"Starting new sweep. Saving to {save_path}...")
        if save_path.exists():
            print(f"Overwriting existing file {save_path}...")
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle Gurobi jobs separately if specified
    if compute_gurobi_separately:
        if gurobi_nodelist is None:
            warn(
                "Gurobi nodelist is None but compute_gurobi_separately is True. "
                "Gurobi jobs will be submitted to the same nodes as the other jobs."
            )
            gurobi_nodelist = submitit_kwargs["slurm_nodelist"]

        # Prepare arguments for Gurobi jobs
        job_kwargs_gurobi = job_kwargs.copy()
        job_kwargs_gurobi["solver"] = [Gurobi()]
        job_kwargs_gurobi["distortion_metric"] = [GromovWasserstein()]
        submitit_kwargs_gurobi = submitit_kwargs.copy()
        submitit_kwargs_gurobi["slurm_nodelist"] = gurobi_nodelist

        # Sweep Gurobi jobs
        jobs_gurobi = sweep(
            function=function,
            job_kwargs=job_kwargs_gurobi,
            num_jobs=num_jobs,
            path_to_logs=path_to_logs,
            runtime_per_function=runtime_per_function,
            cluster=cluster,
            condition=condition,
            shuffle=shuffle,
            max_num_timeout=max_num_timeout,
            submitit_kwargs=submitit_kwargs_gurobi,
        )

        # Sweep non-Gurobi jobs
        jobs_wo_gurobi = sweep(
            function=function,
            job_kwargs=job_kwargs,
            num_jobs=num_jobs,
            path_to_logs=path_to_logs,
            runtime_per_function=runtime_per_function,
            cluster=cluster,
            condition=condition,
            shuffle=shuffle,
            max_num_timeout=max_num_timeout,
            submitit_kwargs=submitit_kwargs,
        )

        # Combine Gurobi and non-Gurobi jobs
        jobs = jobs_gurobi + jobs_wo_gurobi
    else:
        # If not computing Gurobi separately, add Gurobi to the list of solvers and sweep all jobs
        solvers.append(Gurobi())
        jobs = sweep(
            function=function,
            job_kwargs=job_kwargs,
            num_jobs=num_jobs,
            path_to_logs=path_to_logs,
            runtime_per_function=runtime_per_function,
            cluster=cluster,
            condition=condition,
            shuffle=shuffle,
            max_num_timeout=max_num_timeout,
            submitit_kwargs=submitit_kwargs,
        )

    # Monitor the jobs
    monitor_jobs(jobs)

    # load the results
    df = load_df_from_jobs(jobs, force=force)

    # save the results
    df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")
    return df


def plot_cost_bound_size(
    df: pd.DataFrame,
    save_path: Path,
):
    """
    Plots the cost and bound against problem size for different solvers.

    The cost and bound are normalized by the square of the problem size.
    The plot is saved to the specified path.

    Args:
        df: DataFrame containing the experimental results. Must include columns
            'GromovWasserstein()', 'Problem size', 'Bound', and 'solver'.
        save_path: Path to save the generated plot.
    """
    # normalize cost and bound by the square of the problem size
    df["Cost"] = df["GromovWasserstein()"] / df["Problem size"] ** 2
    df["Bound"] = df["Bound"] / df["Problem size"] ** 2

    # Rename solvers for better legend display
    df = rename_solvers(df)

    # Define the order of solvers for consistent plotting
    solver_order = [
        "Random",
        "LocalCKA",
        "OT",
        "FAQ",
        "MPOpt",
        "Gurobi",
        "Ours",
    ]

    fig, ax = plt.subplots(figsize=(10, 6.18))

    # Plot cost lines for each solver
    sns.lineplot(
        data=df,
        x="Problem size",
        y="Cost",
        hue="solver",
        hue_order=solver_order,
        style="solver",
        palette=palette,
        markers=markers,
        dashes=False,
        markersize=markersize,
        ax=ax,
    )
    # Plot bound lines for each solver
    sns.lineplot(
        data=df,
        x="Problem size",
        y="Bound",
        hue="solver",
        hue_order=solver_order,
        style="solver",
        palette=palette,
        markers=markers,
        markersize=10,
        dashes=[dash] * len(solver_order),
        ax=ax,
    )
    ax.set_ylabel("Cost / Bound")

    # Create custom legend handles
    handles = [
        Line2D(
            [0],
            [0],
            color=palette[s],
            marker=markers[s],
            markersize=markersize,
            label=s,
        )
        for s in solver_order
    ]
    handles += [
        Line2D([0], [0], color="black", label="Cost"),
        Line2D([0], [0], color="black", dashes=dash, label="Bound"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.04, 0.5))

    # Save the figure
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # --- Experiment Configuration ---
    # Define experiment name for logging and output paths
    experiment_name = "solver_comparison_larger"

    # Set the number of parallel jobs for sweeps
    num_jobs = 60

    # Define the vision model to be used
    vision_model = CLIP(name="ViT-L/14@336")

    # Define the language model to be used
    language_model = SentenceT(name="all-mpnet-base-v2")

    # Load the dataset
    dataset = CIFAR100(
        root=data_root,
        train=False,
        download=True,
    )

    # Define the primary distortion metric
    distortion_metric = GromovWasserstein()
    # Define the list of solvers to compare (Gurobi is handled separately by sweep_or_load_jobs_separate_gurobi)
    solvers = [
        MPOpt(),
        OptimalTransport(),
        ScipySolver(method="faq", num_repetitions=1),
        ScipySolver(
            method="faq", num_repetitions=1, transpose_first=True
        ),  # This corresponds to LocalCKA
        FactorizedHahnGrant(verbose=True),  # This corresponds to "Ours"
        RandomSolver(),
    ]

    # Define the range of problem sizes (number of samples in subsets)
    subset_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # Parameter for subset selection (e.g., best k subsets)
    best_k = 1

    # Time limits for different stages of the experiment
    timelimit_sec_optimization_subsets = (
        60 * 60
    )  # 1 hour for subset optimization
    timelimit_sec_optimization_matching = (
        5400  # 1.5 hours for matching optimization
    )
    timelimit_min_job = (
        60 * 2
    )  # 2 hours total job time limit, accounting for setup and teardown

    recompute_subsets = False  # Set to True to recompute all subsets
    recompute = False  # Set to True to recompute all jobs

    # --- Embedding Extraction ---
    # Submit jobs to compute missing embeddings if they don't exist
    submit_missing_embeddings(
        path_to_embeddings,
        path_to_logs,
        [vision_model],
        [language_model],
        [dataset],
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

    # --- Subsets ---
    # compute the missing subsets on Gurobi CPU nodes
    compute_missing_subsets(
        vision_models=[vision_model],
        language_models=[language_model],
        datasets=[dataset],
        distortion_metrics=[distortion_metric],
        subset_sizes=subset_sizes,
        best_k=best_k,
        timelimit_sec_optimization=timelimit_sec_optimization_subsets,
        num_jobs=num_jobs,
        path_to_logs=path_to_logs,
        path_to_embeddings=path_to_embeddings,
        path_to_subsets=path_to_subsets,
        timelimit_min_job=timelimit_min_job,
        cluster=cluster,
        nodelist=gurobi_nodelist,
        recompute_subsets=recompute_subsets,
    )

    # Load the precomputed or newly computed subsets
    subsets = load_subsets_from_kwargs(
        vision_model=vision_model,
        language_model=language_model,
        dataset=dataset,
        distortion_metric=distortion_metric,
        subset_sizes=subset_sizes,
        best_k=best_k,
        path_to_subsets=path_to_subsets,
        timelimit=timelimit_sec_optimization_subsets,
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

    # Run the main matching experiment sweep or load existing results
    df = sweep_or_load_jobs_separate_gurobi(
        function=UnsupervisedMatching(),
        job_kwargs={
            "vision_model": [vision_model],
            "language_model": [language_model],
            "dataset": [dataset],
            "solver": solvers,  # List of solvers (Gurobi handled by the sweep function)
            "distortion_metric": get_distortion_metrics,  # Dynamically set distortion metric
            "path_to_embeddings": [path_to_embeddings],
            "num_seeds": [1],  # Number of random seeds for stochastic solvers
            "seed": [seed],  # Global seed
            "subset": subsets,  # The precomputed subsets to run matching on
            "timelimit": [timelimit_sec_optimization_matching],
            "reference_distortion_metrics": [
                [distortion_metric]
            ],  # For consistent evaluation
        },
        num_jobs=num_jobs,
        save_path=path_to_raw_results
        / f"{experiment_name}.csv",  # Path to save raw results
        path_to_logs=path_to_logs,
        runtime_per_function=timelimit_min_job,  # Max runtime per job
        cluster=cluster,
        recompute=recompute,
        condition=None,
        shuffle=True,
        max_num_timeout=0,
        submitit_kwargs={
            "name": experiment_name,
            "nodes": 1,
            "tasks_per_node": 1,
            "cpus_per_task": 4,
            "gpus_per_node": 0,  # Matching is CPU-bound
            "mem_gb": 64,
            "slurm_nodelist": same_cpu_nodelist,  # Default nodelist for non-Gurobi jobs
        },
        force=False,  # Do not force load if jobs failed
        compute_gurobi_separately=True,  # Run Gurobi jobs on specified nodes
        gurobi_nodelist=gurobi_nodelist,  # Nodelist for Gurobi jobs
    )

    # --- Result Processing and Saving ---
    # Define path for processed results
    path_to_these_results = path_to_processed_results / experiment_name
    path_to_these_results.mkdir(parents=True, exist_ok=True)

    # Cost and Bound vs. Problem Size (Fig. 6)
    plot_cost_bound_size(
        df=df,
        save_path=path_to_these_results / "fig_6.png",
    )
