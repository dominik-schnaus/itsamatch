"""
This script conducts the ablations on the Hahn-Grant algorithm (Sec. D.3).

It involves:
1. Running matchings of synthetic data with different variations of the solver.
2. Saving the results in a CSV file (Tab. 4, 5).
"""

from time import time

import pandas as pd
import torch
from submitit.helpers import Checkpointable
from torch.nn.functional import normalize

from itsamatch.distortion_metrics import DistortionMetric, InnerProduct
from itsamatch.experiments.utils import (
    cluster,
    path_to_logs,
    path_to_processed_results,
    path_to_raw_results,
    seed,
)
from itsamatch.solvers import (
    FactorizedHahnGrant,
    HahnGrant,
    QAPSolver,
)
from itsamatch.utils import get_seeds, seed_everything, sweep_or_load_jobs


class HahnGrantAblation(Checkpointable):
    """
    A checkpointable class to run ablation studies on Hahn-Grant solvers.

    This class encapsulates the logic for running a single experiment instance,
    generating data, solving the QAP, and collecting results.
    """

    @torch.inference_mode()
    def __call__(
        self,
        solver: QAPSolver,
        distortion_metric: DistortionMetric,
        size: int,
        dimensionality: int,
        seed: int = 42,
        num_seeds: int = 20,
        which_seed: int | None = None,
        timelimit: int | None = None,
    ):
        """
        Run a single experiment for the Hahn-Grant ablation study.

        Args:
            solver: The QAP solver to use.
            distortion_metric: The distortion metric for cost calculation.
            size: The size of the problem (number of items to match).
            dimensionality: The dimensionality of the feature vectors.
            seed: The global random seed for generating multiple local seeds.
            num_seeds: The total number of local seeds to generate from the global seed.
            which_seed: If specified, runs the experiment for only this specific local seed index.
            timelimit: Time limit for the solver in seconds.

        Returns:
            A pandas DataFrame containing the results of the experiment(s).
        """
        seeds = get_seeds(seed, num_seeds)

        if which_seed is not None:
            seeds = [seeds[which_seed]]  # Select a specific seed if requested

        results = []

        for local_seed in seeds:
            seed_everything(
                local_seed
            )  # Ensure reproducibility for this iteration
            # Generate random source and target vectors
            source_vector = torch.randn(size, dimensionality)
            target_vector = torch.randn(size, dimensionality)

            # Normalize the vectors
            source_vector = normalize(source_vector, dim=1)
            target_vector = normalize(target_vector, dim=1)

            # Compute the kernel matrices based on the distortion metric
            source_kernel_matrix = distortion_metric.kernel_source(
                source_vector
            )
            target_kernel_matrix = distortion_metric.kernel_target(
                target_vector
            )

            start_time = time()

            # Solve the assignment problem
            if solver.factorized:
                # For factorized solvers, compute costs and constant separately
                cost1 = -distortion_metric.loss.h1(source_kernel_matrix)
                cost2 = distortion_metric.loss.h2(target_kernel_matrix)
                constant = (
                    distortion_metric.loss.f1(source_kernel_matrix).sum()
                    + distortion_metric.loss.f2(target_kernel_matrix).sum()
                )
                prediction, optimal, cost, bound = solver.solve(
                    cost1=cost1,
                    cost2=cost2,
                    constant=constant,
                    timelimit=timelimit,
                )
            else:
                # For non-factorized solvers, compute the full cost matrix
                full_cost_matrix = distortion_metric.loss(
                    source_kernel_matrix[:, None, :, None],
                    target_kernel_matrix[None, :, None, :],
                )
                prediction, optimal, cost, bound = solver.solve(
                    full_cost_matrix, timelimit=timelimit
                )

            end_time = time()
            time_elapsed = end_time - start_time

            # Store results for this iteration
            results_dict = {
                "local_seed": local_seed,
                "prediction": prediction.tolist(),
                "Optimal": optimal,
                "Cost": cost,
                "Bound": bound,
                "Time elapsed": time_elapsed,
            }

            results.append(results_dict)

        # Convert results to a DataFrame and add experiment parameters
        df = pd.DataFrame(results)
        df["global_seed"] = seed
        df["Problem size"] = size
        df["dimensionality"] = dimensionality
        df["solver"] = repr(solver)
        df["distortion_metric"] = str(distortion_metric)
        df["timelimit"] = timelimit
        df["which_seed"] = which_seed

        return df


if __name__ == "__main__":
    # --- Experiment Configuration ---
    experiment_name = "hahn_grant_ablation"

    num_jobs = 30  # Number of parallel jobs for sweep
    num_seeds = 5  # Number of local seeds per (solver, size) combination
    sizes = [40, 100]  # Problem sizes to test
    timelimit_min_job = int(2.5 * 60)  # Minimum runtime per job in sweep

    recompute = False  # Whether to recompute results or load existing ones

    # Define the solver configurations to compare
    solvers = {
        "Hahn-Grant": HahnGrant(),
        "+ factorized": FactorizedHahnGrant(
            matching_algorithm="hungarian",
            initialize_with_2opt_faq=False,
            update_with_lap_solutions=False,
        ),
        "+ auction": FactorizedHahnGrant(
            matching_algorithm="forward_backward_auction_cpp",
            initialize_with_2opt_faq=False,
            update_with_lap_solutions=False,
        ),
        "+ Jonker-Volgenant": FactorizedHahnGrant(
            matching_algorithm="lapjv",
            initialize_with_2opt_faq=False,
            update_with_lap_solutions=False,
        ),
        "+ LAP solutions": FactorizedHahnGrant(
            matching_algorithm="lapjv",
            initialize_with_2opt_faq=False,
            update_with_lap_solutions=True,
        ),
        "+ primal heuristic": FactorizedHahnGrant(
            matching_algorithm="lapjv",
            initialize_with_2opt_faq=True,
            update_with_lap_solutions=True,
        ),
    }

    # Compute the bounds and costs for different solvers and sizes
    def get_timelimit(size: int, **kwargs) -> list[int]:
        """
        Determine the timelimit for the solver based on the problem size.

        Args:
            size: The problem size.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A list containing the timelimit in seconds.

        Raises:
            ValueError: If the size is not recognized.
        """
        if size == 40:
            return [60 * 60]  # 1 hour
        elif size == 100:
            return [2 * 60 * 60]  # 2 hours
        else:
            raise ValueError(f"Unknown size: {size}")

    # --- Matching ---
    # Run the experiments using sweep_or_load_jobs
    df = sweep_or_load_jobs(
        function=HahnGrantAblation(),
        job_kwargs={
            "solver": list(solvers.values()),
            "distortion_metric": [InnerProduct()],
            "size": sizes,
            "dimensionality": [1024],
            "seed": [
                seed
            ],  # Global seed for reproducibility of seed generation
            "num_seeds": [
                num_seeds
            ],  # Number of local seeds to run per job_kwarg combination
            "which_seed": list(
                range(num_seeds)
            ),  # Distribute local seeds across jobs
            "timelimit": get_timelimit,  # Dynamically set timelimit based on size
        },
        num_jobs=num_jobs,
        save_path=path_to_raw_results / f"{experiment_name}.csv",
        path_to_logs=path_to_logs,
        runtime_per_function=timelimit_min_job,
        cluster=cluster,
        recompute=recompute,
        condition=None,
        shuffle=True,  # Shuffle job order
        max_num_timeout=0,
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

    # Compare the different variations (Tab. 4, 5)
    # Normalize cost and bound by problem size squared
    df["Cost"] = df["Cost"] / df["Problem size"] ** 2
    df["Bound"] = df["Bound"] / df["Problem size"] ** 2

    # Map solver representations to human-readable names
    solver_to_name = {}
    for key, solver_obj in solvers.items():
        solver_to_name[repr(solver_obj)] = key

    df["Solver"] = df["solver"].map(lambda s: solver_to_name[s])

    # Group results by problem size and solver, then calculate mean cost and bound
    out = (
        df.groupby(["Problem size", "Solver"])[["Cost", "Bound"]]
        .mean()
        .loc[
            (sizes, list(solvers.keys())), :
        ]  # Ensure correct order of solvers and sizes
    )
    # Save the processed results
    out.to_csv(path_to_these_results / "tab_4_5.csv")
