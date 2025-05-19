"""
Utility module for experiment management, including job submission, parameter sweeping,
and result handling with support for distributed computing environments.
"""

import pickle
import random
from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import submitit
import torch
from submitit.helpers import FunctionSequence
from tqdm import tqdm


def get_seeds(
    seed: int | None = None, num_seeds: int | None = None
) -> list[int]:
    """Generate a list of random seeds from an initial seed.

    Args:
        seed: Initial seed to initialize the random generator
        num_seeds: Number of seeds to generate

    Returns:
        A list of randomly generated seeds
    """
    if seed is None:
        seed = torch.initial_seed()

    if num_seeds is None:
        num_seeds = 1

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    # Generate random integers up to the maximum uint32 value
    return torch.randint(
        np.iinfo(np.uint32).max,
        (num_seeds,),
        dtype=torch.long,
        generator=generator,
    ).tolist()


def seed_everything(seed: int | None = None) -> int:
    r"""Adapted from PyTorch Lightning

    Function that sets the seed for pseudo-random number generators in: torch, numpy, and Python's random module.
    In addition, sets the following environment variables:

    - ``PL_GLOBAL_SEED``: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - ``PL_SEED_WORKERS``: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If ``None``, it will read the seed from ``PL_GLOBAL_SEED`` env variable. If ``None`` and the
            ``PL_GLOBAL_SEED`` env variable is not set, then the seed defaults to 0.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~lightning.fabric.utilities.seed.pl_worker_init_function`.

    Returns:
        The seed value used for initialization
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        seed = 0
    elif not isinstance(seed, int):
        seed = int(seed)

    # Ensure seed is within valid bounds
    if not (min_seed_value <= seed <= max_seed_value):
        print(
            f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = 0

    # Set seeds for Python's random, numpy, and torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def submit_grouped(
    functions: list[partial],
    num_jobs: int,
    path_to_logs: Path,
    runtime_per_function: int,
    cluster: str,
    shuffle: bool = True,
    max_num_timeout: int = 3,
    **submitit_kwargs,
):
    """Submit a list of functions as grouped jobs to a cluster.

    Args:
        functions: List of partial functions to submit as jobs
        num_jobs: Number of job groups to create (functions will be divided among these)
        path_to_logs: Path where logs and snapshots will be stored
        runtime_per_function: Estimated runtime for each function in minutes
        cluster: Cluster type ('slurm', 'local', 'debug')
        shuffle: Whether to shuffle functions before grouping
        max_num_timeout: Maximum number of timeouts for cluster jobs
        **submitit_kwargs: Additional arguments for submitit executor

    Returns:
        List of submitted jobs
    """
    # Calculate the total runtime needed per job group
    total_runtime_per_job = max(
        (runtime_per_function * len(functions) + 1) // num_jobs,
        runtime_per_function,
    )
    snapshot_dir = path_to_logs / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    with submitit.helpers.RsyncSnapshot(
        snapshot_dir=snapshot_dir
    ):  # to save the logs
        # Configure cluster-specific parameters
        if cluster != "debug":
            cluster_specific_kwargs = {
                f"{cluster}_max_num_timeout": max_num_timeout,
            }
        else:
            cluster_specific_kwargs = {}
        # Initialize executor with appropriate parameters
        executor = submitit.AutoExecutor(
            folder=path_to_logs, cluster=cluster, **cluster_specific_kwargs
        )
        executor.update_parameters(**submitit_kwargs)
        executor.update_parameters(timeout_min=total_runtime_per_job)
        jobs = []

        if shuffle:
            # Shuffle the functions to avoid bias in the grouping
            permutation = torch.randperm(len(functions)).tolist()
            functions = [functions[i] for i in permutation]

        # Submit jobs in batches
        with executor.batch():
            # Group functions and submit each group as a job
            for i in range(num_jobs):
                start = i * len(functions) // num_jobs
                end = (i + 1) * len(functions) // num_jobs
                if end - start >= 1:
                    fn = FunctionSequence(verbose=True)
                    for j in range(start, end):
                        fn.add(functions[j])
                    job = executor.submit(fn)
                    jobs.append(job)
            print(
                f"Submitting {len(functions)} jobs grouped into {num_jobs} grouped jobs..."
            )
    return jobs


def sweep(
    function: Callable,
    job_kwargs: dict[str, list | Callable[[Any], list]],
    num_jobs: int,
    path_to_logs: Path,
    runtime_per_function: int,
    cluster: str,
    condition: Callable | None = None,
    shuffle: bool = True,
    max_num_timeout: int = 3,
    submitit_kwargs: dict | None = None,
):
    """Perform a parameter sweep by generating and submitting jobs for different parameter combinations.

    Args:
        function: The base function to call for each parameter combination
        job_kwargs: Dictionary of parameters to sweep over, values can be lists or callables
            if callable, it should return a list of values based on the current kwargs
        num_jobs: Number of job groups to create
        path_to_logs: Path where logs will be stored
        runtime_per_function: Estimated runtime per function in minutes
        cluster: Cluster type for job submission
        condition: Optional callable to filter parameter combinations
        shuffle: Whether to shuffle functions before grouping
        max_num_timeout: Maximum number of timeouts for cluster jobs
        submitit_kwargs: Additional arguments for submitit

    Returns:
        List of submitted jobs
    """
    # Separate kwargs into those with fixed lists and those with callables
    callable_kwargs = {
        k: v for k, v in job_kwargs.items() if not isinstance(v, list)
    }
    list_kwargs = {k: v for k, v in job_kwargs.items() if isinstance(v, list)}

    # Generate the cartesian product of list parameters
    ordered_keys, ordered_values = zip(*list_kwargs.items())
    sweeps = list(product(*ordered_values))

    functions = []
    for combination in sweeps:
        kwargs = dict(zip(ordered_keys, combination))

        # Generate values from callable parameters based on current kwargs
        callable_values = [v(**kwargs) for v in callable_kwargs.values()]

        # Generate additional combinations for callable parameters
        new_sweep = list(product(*callable_values))
        for new_combination in new_sweep:
            new_kwargs = dict(zip(callable_kwargs.keys(), new_combination))
            kwargs.update(new_kwargs)

            # Apply condition filter if provided
            if condition is not None:
                if not condition(**kwargs):
                    continue

            # Create partial function with the parameter combination
            function_instance = partial(
                deepcopy(function),
                **kwargs,
            )
            functions.append(function_instance)

    if len(functions) == 0:
        return []

    # Submit the generated functions as grouped jobs
    jobs = submit_grouped(
        functions=functions,
        num_jobs=num_jobs,
        path_to_logs=path_to_logs,
        runtime_per_function=runtime_per_function,
        cluster=cluster,
        shuffle=shuffle,
        max_num_timeout=max_num_timeout,
        **(submitit_kwargs or {}),
    )

    # Print first job ID for reference
    print(f"Job id: {jobs[0].job_id}")
    return jobs


def monitor_jobs(
    jobs: list[submitit.Job],
):
    """Wait for all submitted jobs to complete and handle any exceptions.

    Args:
        jobs: List of submitit job objects to monitor
    """
    # Wait for all jobs to finish, showing progress
    for job in tqdm(
        submitit.helpers.as_completed(jobs),
        desc="Waiting for jobs to finish",
        total=len(jobs),
    ):
        try:
            job.result()
        except Exception as e:
            print(f"Job failed: {e}")


def load_df_from_jobs(
    jobs: list[submitit.Job],
    force: bool = True,
):
    """Load and combine results from completed jobs into a single DataFrame.

    Args:
        jobs: List of submitit job objects with results to load
        force: If True, raises exceptions from failed jobs; otherwise prints warnings

    Returns:
        Combined pandas DataFrame with results from all jobs
    """
    dfs = []
    for job in jobs:
        try:
            # Get job result (expected to be DataFrame or list of DataFrames)
            df = job.result()
        except Exception as e:
            if force:
                raise e
            print(f"Job {job.job_id} failed: {e}")
            continue

        # Collect results, handling both single DataFrame and list of DataFrames
        if isinstance(df, pd.DataFrame):
            dfs.append(df)
        else:
            dfs.extend(df)

    # Combine all results into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)
    return df


def sweep_or_load_jobs(
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
):
    """Either load results from an existing file or perform a parameter sweep.

    Args:
        function: The base function to call for each parameter combination
        job_kwargs: Dictionary of parameters to sweep over
        num_jobs: Number of job groups to create
        save_path: Path to save/load results
        path_to_logs: Path where logs will be stored
        runtime_per_function: Estimated runtime per function in minutes
        cluster: Cluster type for job submission
        recompute: Whether to recompute results even if save_path exists
        condition: Optional callable to filter parameter combinations
        shuffle: Whether to shuffle functions before grouping
        max_num_timeout: Maximum number of timeouts for cluster jobs
        submitit_kwargs: Additional arguments for submitit
        force: Whether to raise exceptions from failed jobs when loading results

    Returns:
        DataFrame with results from the sweep or loaded from file
    """
    # Check if results already exist and should be loaded
    if save_path.exists() and not recompute:
        print(f"Path {save_path} already exists. Loading file...")
        return pd.read_csv(save_path)
    else:
        print(f"Starting new sweep. Saving to {save_path}...")
        if save_path.exists():
            print(f"Overwriting existing file {save_path}...")
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Perform parameter sweep
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

    # Monitor jobs until completion
    monitor_jobs(jobs)

    # Load and combine results
    df = load_df_from_jobs(jobs, force=force)

    # Save final results
    df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")
    return df


def save_results_from_jobid(
    job_id: str | int,
    path_to_logs: Path,
    save_path: Path,
):
    """Load and save results from a specific job ID.

    Useful for recovering results when the main script was interrupted.

    Args:
        job_id: The ID of the job to recover results from
        path_to_logs: Path where job result files are stored
        save_path: Path to save the combined results
    """
    print(f"Loading results for id {job_id}...")
    dfs = []

    # Scan log directory for result files matching the job ID
    for file in tqdm(list(path_to_logs.iterdir())):
        if file.name.endswith("_result.pkl") and file.name.startswith(
            f"{job_id}_"
        ):
            # Load results from pickle file
            with open(file, "rb") as f:
                state, df = pickle.load(f)
            if state == "success":
                dfs.extend(df)
                assert isinstance(df, list), (
                    f"Expected DataFrame, got {type(df)}"
                )
            else:
                print(f"Job {file.name} failed with state {state}.")

    # Combine and save results
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(save_path, index=False)
