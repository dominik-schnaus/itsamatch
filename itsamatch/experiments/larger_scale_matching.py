"""
This script conducts larger-scale matching experiments (Sec. 5.2).

It involves:
1. Extracting embeddings for various vision and language model combinations on specified datasets.
2. Computing optimal subsets of different sizes using Gurobi.
3. Computing matching accuracies for the different subsets.
4. Generating and saving plots of matching accuracies against problem sizes (Fig. 5).
"""

from functools import partial
from pathlib import Path
from time import time
from warnings import warn

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from submitit.helpers import Checkpointable
from torch.nn.functional import normalize
from tqdm import tqdm

try:
    import gurobipy as gp
    from gurobipy import GRB

    gurobi_available = True
except ImportError:
    warn(
        "Gurobi is not installed. Please install it with 'pip install gurobipy' and set up your Gurobi license."
        " See https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license for more information."
    )
    gurobi_available = False


from itsamatch.datasets import (
    CIFAR100,
    Dataset,
    ImageNet100,
)
from itsamatch.distortion_metrics import (
    DistortionMetric,
    GromovWasserstein,
)
from itsamatch.experiments.utils import (
    cluster,
    data_root,
    gurobi_nodelist,
    imagenet_root,
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
from itsamatch.extract_embeddings import (
    encode_prompts,
    load_and_preprocess_embeddings,
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
    VisionModel,
)
from itsamatch.solvers import FactorizedHahnGrant, QAPSolver
from itsamatch.utils import (
    get_seeds,
    monitor_jobs,
    seed_everything,
    sweep,
    sweep_or_load_jobs,
)


def standardize_kernel_matrix(kernel_matrix):
    """
    Standardize a kernel matrix by subtracting the mean and dividing by standard deviation.

    The diagonal elements are ignored and set to 0 after standardization.

    Args:
        kernel_matrix: Tensor containing kernel values

    Returns:
        Standardized kernel matrix
    """
    all_but_diagonal = torch.ones_like(
        kernel_matrix, dtype=torch.bool
    ).fill_diagonal_(False)
    kernel_standardized = (
        (kernel_matrix - kernel_matrix[all_but_diagonal].mean())
        / kernel_matrix[all_but_diagonal].std()
    ).fill_diagonal_(0)
    return kernel_standardized


class UnsupervisedMatching(Checkpointable):
    """
    Performs unsupervised matching between vision and language embeddings.

    This class implements the core matching algorithm that aligns vision embeddings with
    language embeddings using a given distortion metric and quadratic assignment solver.
    """

    @torch.inference_mode()
    def __call__(
        self,
        vision_model: VisionModel,
        language_model: LanguageModel,
        dataset: Dataset,
        solver: QAPSolver,
        distortion_metric: DistortionMetric,
        path_to_embeddings: Path,
        num_seeds: int = 20,
        seed: int = 42,
        normalize_before_averaging: bool = False,
        standardize_kernel: bool = True,
        subset: list[bool] = None,
        which_seed: int | None = None,
        timelimit: int | None = None,
        reference_distortion_metrics: list[DistortionMetric] | None = None,
    ):
        """
        Run an unsupervised matching experiment.

        Args:
            vision_model: The vision model to use for extracting embeddings
            language_model: The language model to use for extracting embeddings
            dataset: The dataset containing images and labels
            solver: QAP solver to use for matching
            distortion_metric: Metric to use for measuring distortion
            path_to_embeddings: Path to pre-computed embeddings
            num_seeds: Number of seeds to use for randomization
            seed: Base seed for random number generation
            normalize_before_averaging: Whether to normalize embeddings before averaging
            standardize_kernel: Whether to standardize kernel matrices
            subset: Optional subset of classes to consider
            which_seed: Specific seed index to run (if None, runs all seeds)
            timelimit: Time limit for solver in seconds
            reference_distortion_metrics: Optional list of reference metrics

        Returns:
            DataFrame containing matching results
        """
        seeds = get_seeds(seed, num_seeds)

        if which_seed is not None:
            seeds = [seeds[which_seed]]

        if reference_distortion_metrics is None:
            reference_distortion_metrics = []

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

        # compute language kernel matrix
        language_kernel_matrix = distortion_metric.kernel_target(
            language_embeddings
        )

        if standardize_kernel:
            # standardize kernel matrix
            language_kernel_matrix = standardize_kernel_matrix(
                language_kernel_matrix
            )

        # number of samples and kernel matrices
        num_samples = vision_embeddings.shape[0]

        results = []

        for local_seed in tqdm(seeds):
            seed_everything(local_seed)

            sampled_indices = torch.randperm(
                num_samples,
                generator=torch.Generator().manual_seed(local_seed),
            )[: num_samples // 2]
            vision_embeddings_subsampled = vision_embeddings[sampled_indices]
            labels_subsampled = labels[sampled_indices]

            # average vision embeddings
            vision_embeddings_averaged = torch.stack(
                [
                    vision_embeddings_subsampled[labels_subsampled == i].mean(
                        dim=0
                    )
                    for i in labels.unique(sorted=True)
                ]
            )

            if subset is not None:
                vision_embeddings_averaged = vision_embeddings_averaged[subset]
                language_kernel_matrix = language_kernel_matrix[subset][
                    :, subset
                ]

            # normalize vision embeddings
            vision_embeddings_norm = normalize(
                vision_embeddings_averaged, dim=-1
            )

            # compute vision kernel matrix
            vision_kernel_matrix = distortion_metric.kernel_source(
                vision_embeddings_norm
            )

            if standardize_kernel:
                # standardize kernel matrix
                vision_kernel_matrix = standardize_kernel_matrix(
                    vision_kernel_matrix
                )

            start_time = time()

            if solver.factorized:
                cost1 = -distortion_metric.loss.h1(vision_kernel_matrix)
                cost2 = distortion_metric.loss.h2(language_kernel_matrix)
                constant = (
                    distortion_metric.loss.f1(vision_kernel_matrix).sum()
                    + distortion_metric.loss.f2(language_kernel_matrix).sum()
                )
                prediction, optimal, cost, bound = solver.solve(
                    cost1=cost1,
                    cost2=cost2,
                    constant=constant,
                    timelimit=timelimit,
                )
            else:
                full_cost_matrix = distortion_metric.loss(
                    vision_kernel_matrix[:, None, :, None],
                    language_kernel_matrix[None, :, None, :],
                )
                prediction, optimal, cost, bound = solver.solve(
                    full_cost_matrix, timelimit=timelimit
                )

            end_time = time()
            time_elapsed = end_time - start_time

            target = torch.arange(vision_kernel_matrix.shape[0])
            accuracy = (prediction == target).float().mean()

            # collect results
            results_dict = {
                "Matching accuracy": accuracy.item(),
                "local_seed": local_seed,
                "prediction": prediction.tolist(),
                "Optimal": optimal,
                "Cost": cost,
                "Bound": bound,
                "Time elapsed": time_elapsed,
            }
            for reference_distortion_metric in reference_distortion_metrics:
                reference_cost = (
                    reference_distortion_metric.loss(
                        vision_kernel_matrix,
                        language_kernel_matrix[prediction][:, prediction],
                    )
                    .sum()
                    .item()
                )
                results_dict[repr(reference_distortion_metric)] = reference_cost
            results.append(results_dict)

        # build DataFrame and save
        df = pd.DataFrame(results)
        df["global_seed"] = seed
        df["num_samples"] = num_samples
        df["num_seeds"] = num_seeds
        df["vision_model_class"] = vision_model.__class__.__name__
        df["vision_model_name"] = vision_model.name
        df["language_model_class"] = language_model.__class__.__name__
        df["language_model_name"] = language_model.name
        df["dataset"] = str(dataset)
        df["distortion_metric"] = str(distortion_metric)
        df["solver"] = str(solver)
        df["subset"] = str(subset) if subset is not None else None
        df["Problem size"] = (
            sum(subset) if subset is not None else language_embeddings.shape[0]
        )
        df["timelimit"] = timelimit
        df["which_seed"] = which_seed
        df["normalize_before_averaging"] = normalize_before_averaging
        return df


def get_subset(cost, subset_size, best_k, timelimit):
    """
    Solve the subset selection (p-dispersion-sum) problem using Gurobi.

    Finds the optimal subset of a given size that minimizes the quadratic cost function.

    Args:
        cost: Cost matrix
        subset_size: Size of subset to select
        best_k: Number of best solutions to find
        timelimit: Time limit for optimization in seconds

    Returns:
        Tuple of (list of selected subsets, list of objective values)
    """
    if not gurobi_available:
        raise ImportError(
            "Gurobi is not installed. Please install it with 'pip install gurobipy' and set up your Gurobi license."
            " See https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license for more information."
        )

    # Create a new model
    m = gp.Model("quadratic assignment")
    m.Params.OutputFlag = 0

    m.Params.SolutionNumber = best_k
    m.Params.PoolSearchMode = 2

    m.Params.TuneCriterion = 2
    m.Params.CliqueCuts = 2
    m.Params.Heuristics = 0.5
    if isinstance(cost, torch.Tensor):
        cost = cost.cpu().numpy()
    shape = len(cost)
    cost = cost + cost.T
    # Create variables
    B = m.addMVar(shape=(shape,), lb=0.0, ub=1.0, vtype=GRB.BINARY, name="B")
    # Set objective
    m.setObjective(
        gp.quicksum(
            cost[i, j] * B[i] * B[j] for i in range(shape) for j in range(shape)
        ),
        GRB.MINIMIZE,
    )
    # Add constraints
    m.addConstr(B.sum() == subset_size, name="b")

    if timelimit is not None:
        m.Params.TimeLimit = timelimit

    # Optimize model
    m.optimize()

    predictions = []
    objective_values = []
    bounds = []

    # is_optimal = m.status == GRB.OPTIMAL

    for i in range(m.SolCount):
        m.Params.SolutionNumber = i
        indices = B.Xn.argsort()[-subset_size:]
        predictions.append([i in indices for i in range(shape)])
        objective_values.append(m.PoolObjVal)
        bounds.append(m.PoolObjBound)

    # close model
    m.dispose()
    return predictions, objective_values


def get_subset_path(
    vision_model: VisionModel,
    language_model: LanguageModel,
    dataset: Dataset,
    distortion_metric: DistortionMetric,
    subset_size: int,
    best_k: int,
    timelimit: int | None,
    path_to_subsets: Path,
):
    """
    Generate a file path for saving/loading subset results.

    Args:
        vision_model: Vision model used
        language_model: Language model used
        dataset: Dataset used
        distortion_metric: Distortion metric used
        subset_size: Size of subset
        best_k: Index of the solution in the solution pool
        timelimit: Time limit used for optimization
        path_to_subsets: Root path for subset results

    Returns:
        Path object for the subset file
    """
    file_name = f"{vision_model}_{language_model}_{subset_size}_{best_k}_{timelimit}.pt"
    file_name = file_name.replace("/", "").replace(" ", "-").lower()

    distortion_metric_repr = repr(distortion_metric)
    distortion_metric_repr = distortion_metric_repr.replace("(", "_").replace(")", "")

    dataset_repr = f"{dataset}_{encode_prompts(dataset.prompts)}"
    save_path = (
        path_to_subsets
        / distortion_metric_repr
        / dataset_repr
        / file_name
    )
    return save_path


class SubsetJob(Checkpointable):
    """
    Compute and save optimal subsets for matching experiments.

    This class handles the computation of optimal subsets for a given problem
    configuration and saves the results to disk.
    """

    def __call__(
        self,
        vision_model: VisionModel,
        language_model: LanguageModel,
        dataset: Dataset,
        distortion_metric: DistortionMetric,
        subset_size: int,
        best_k: int,
        path_to_embeddings: Path,
        path_to_subsets: Path,
        normalize_before_averaging: bool = False,
        timelimit: int | None = None,
    ):
        """
        Compute optimal subsets and save results.

        Args:
            vision_model: Vision model to use
            language_model: Language model to use
            dataset: Dataset to use
            distortion_metric: Distortion metric to use
            subset_size: Size of subset to select
            best_k: Number of best solutions to find
            path_to_embeddings: Path to pre-computed embeddings
            path_to_subsets: Path to save subset results
            normalize_before_averaging: Whether to normalize embeddings before averaging
            timelimit: Time limit for optimization in seconds

        Returns:
            None
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

        # standardize kernel matrices
        vision_kernel_matrix = standardize_kernel_matrix(vision_kernel_matrix)
        language_kernel_matrix = standardize_kernel_matrix(
            language_kernel_matrix
        )

        # compute cost matrix
        cost = distortion_metric.loss(
            vision_kernel_matrix, language_kernel_matrix
        )

        # get subset
        if subset_size == language_embeddings.shape[0]:
            objective_values = [cost.sum().item()] * best_k
            subsets = [[True] * subset_size] * best_k
        else:
            subsets, objective_values = get_subset(
                cost=cost,
                subset_size=subset_size,
                best_k=best_k,
                timelimit=timelimit,
            )

        for k, (subset, objective_value) in enumerate(
            zip(subsets, objective_values)
        ):
            save_path = get_subset_path(
                vision_model=vision_model,
                language_model=language_model,
                dataset=dataset,
                distortion_metric=distortion_metric,
                subset_size=subset_size,
                best_k=k,
                timelimit=timelimit,
                path_to_subsets=path_to_subsets,
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "subset": subset,
                    "objective_value": objective_value,
                    "vision_model": repr(vision_model),
                    "language_model": repr(language_model),
                    "dataset": repr(dataset),
                    "distortion_metric": repr(distortion_metric),
                    "subset_size": subset_size,
                    "best_k": k,
                    "timelimit": timelimit,
                    "normalize_before_averaging": normalize_before_averaging,
                },
                save_path,
            )
        return


def load_subsets_from_kwargs(
    vision_model: VisionModel,
    language_model: LanguageModel,
    dataset: Dataset,
    distortion_metric: DistortionMetric,
    subset_sizes: list[int],
    best_k: int,
    timelimit: int | None,
    path_to_subsets: Path,
    **kwargs,
):
    """
    Load pre-computed subsets based on experiment parameters.

    Args:
        vision_model: Vision model used
        language_model: Language model used
        dataset: Dataset used
        distortion_metric: Distortion metric used
        subset_sizes: List of subset sizes to load
        best_k: Number of best solutions per subset size
        timelimit: Time limit used for optimization
        path_to_subsets: Root path for subset results
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        List of loaded subsets
    """
    subsets = []
    for subset_size in subset_sizes:
        for k in range(best_k):
            # load subsets
            save_path = get_subset_path(
                vision_model=vision_model,
                language_model=language_model,
                dataset=dataset,
                distortion_metric=distortion_metric,
                subset_size=subset_size,
                best_k=k,
                timelimit=timelimit,
                path_to_subsets=path_to_subsets,
            )

            if save_path.exists():
                data = torch.load(save_path, weights_only=False)
                subsets.append(data["subset"])
            else:
                raise FileNotFoundError(f"Subsets not found at {save_path}")
    return subsets


def compute_subset_condition(
    recompute_subsets: bool,
    vision_model: VisionModel,
    language_model: LanguageModel,
    dataset: Dataset,
    distortion_metric: DistortionMetric,
    subset_size: int,
    best_k: int,
    path_to_subsets: Path,
    timelimit: int | None,
    **kwargs,
) -> bool:
    """
    Check if subset needs to be computed.

    Args:
        recompute_subsets: Flag to force recomputation
        vision_model: Vision model to use
        language_model: Language model to use
        dataset: Dataset to use
        distortion_metric: Distortion metric to use
        subset_size: Size of subset
        best_k: Index of solution
        path_to_subsets: Path to save subset results
        timelimit: Time limit for optimization

    Returns:
        Boolean indicating whether subset computation is needed
    """
    save_path = get_subset_path(
        vision_model=vision_model,
        language_model=language_model,
        dataset=dataset,
        distortion_metric=distortion_metric,
        subset_size=subset_size,
        best_k=best_k,
        timelimit=timelimit,
        path_to_subsets=path_to_subsets,
    )
    return not save_path.exists() or recompute_subsets


def compute_missing_subsets(
    vision_models: list[VisionModel],
    language_models: list[LanguageModel],
    datasets: list[Dataset],
    distortion_metrics: list[DistortionMetric],
    subset_sizes: list[int],
    best_k: int,
    timelimit_sec_optimization: int | None,
    num_jobs: int,
    path_to_logs: Path,
    path_to_embeddings: Path,
    path_to_subsets: Path,
    timelimit_min_job: int | None,
    cluster: str,
    nodelist: str,
    recompute_subsets: bool = False,
):
    """
    Compute missing subsets in parallel using submitit.

    Args:
        vision_models: List of vision models
        language_models: List of language models
        datasets: List of datasets
        distortion_metrics: List of distortion metrics
        subset_sizes: List of subset sizes
        best_k: Number of best solutions to find
        timelimit_sec_optimization: Time limit for optimization in seconds
        num_jobs: Maximum number of parallel jobs
        path_to_logs: Path to log directory
        path_to_embeddings: Path to pre-computed embeddings
        path_to_subsets: Path to save subset results
        timelimit_min_job: Time limit for each job in minutes
        cluster: Cluster configuration to use
        nodelist: List of nodes to use
        recompute_subsets: Whether to recompute existing subsets

    Returns:
        None
    """
    jobs = sweep(
        function=SubsetJob(),
        job_kwargs={
            "vision_model": vision_models,
            "language_model": language_models,
            "dataset": datasets,
            "distortion_metric": distortion_metrics,
            "subset_size": subset_sizes,
            "best_k": [best_k],
            "path_to_embeddings": [path_to_embeddings],
            "path_to_subsets": [path_to_subsets],
            "timelimit": [timelimit_sec_optimization],
        },
        num_jobs=num_jobs,
        path_to_logs=path_to_logs,
        runtime_per_function=timelimit_min_job,
        cluster=cluster,
        condition=partial(
            compute_subset_condition, recompute_subsets=recompute_subsets
        ),
        shuffle=True,
        max_num_timeout=0,
        submitit_kwargs={
            "name": "subset",
            "nodes": 1,
            "tasks_per_node": 1,
            "cpus_per_task": 4,
            "gpus_per_node": 0,
            "mem_gb": 64,
            "slurm_nodelist": nodelist,
        },
    )
    if len(jobs) > 0:
        monitor_jobs(jobs)


def plot_accuracy_size(
    df: pd.DataFrame,
    save_path: Path,
    title: str,
) -> None:
    """
    Plot matching accuracy vs. problem size.

    Args:
        df: DataFrame containing results
        save_path: Path to save the plot
        title: Title for the plot

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 6.18))

    hue_order = [
        vision_model.__class__.__name__ for vision_model in vision_models
    ]
    sns.lineplot(
        data=df,
        x="Problem size",
        y="Matching accuracy",
        errorbar="sd",
        markers=markers,
        hue="vision_model_class",
        hue_order=hue_order,
        palette=palette,
        style="vision_model_class",
        dashes=False,
        markersize=markersize,
        ax=ax,
    )

    # Random baseline
    sizes = df["Problem size"].drop_duplicates().sort_values().to_numpy()
    mean = 1 / sizes
    std = 1 / sizes
    lower = mean - std
    upper = mean + std
    random_label = "Random"
    ax.plot(
        sizes,
        mean,
        label=random_label,
        marker=markers[random_label],
        color=palette[random_label],
        markersize=10,
    )
    ax.fill_between(sizes, lower, upper, alpha=0.2, color="black")

    ax.legend(loc="upper right")
    ax.set_xlabel("Problem size")
    ax.set_ylabel("Matching accuracy")

    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlim(10, 100)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # --- Experiment Configuration ---
    experiment_name = "larger_scale_matching"

    num_jobs = 30

    vision_models = [
        CLIP(name="ViT-L/14@336"),
        DeiT(name="DeiT-B/16d@384"),
        DINOv2(name="ViT-G/14"),
    ]

    language_models = [
        SentenceT(name="all-mpnet-base-v2"),
    ]

    datasets = [
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

    distortion_metric = GromovWasserstein()
    solver = FactorizedHahnGrant()

    num_seeds = 3
    best_k = 10

    subset_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    recompute_subsets = False
    recompute_matching = False

    timelimit_sec_optimization = 60 * 60  # 1 hour
    timelimit_min_job = (
        60 * 2
    )  # 2 hours because the jobs need some time to start and finish

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

    # --- Subsets ---
    # compute the missing subsets on Gurobi CPU nodes
    compute_missing_subsets(
        vision_models=vision_models,
        language_models=language_models,
        datasets=datasets,
        distortion_metrics=[distortion_metric],
        subset_sizes=subset_sizes,
        best_k=best_k,
        timelimit_sec_optimization=timelimit_sec_optimization,
        num_jobs=num_jobs,
        path_to_logs=path_to_logs,
        path_to_embeddings=path_to_embeddings,
        path_to_subsets=path_to_subsets,
        timelimit_min_job=timelimit_min_job,
        cluster=cluster,
        nodelist=gurobi_nodelist,
        recompute_subsets=recompute_subsets,
    )

    # --- Matching ---
    subset_fn = partial(
        load_subsets_from_kwargs,
        subset_sizes=subset_sizes,
        best_k=best_k,
        timelimit=timelimit_sec_optimization,
        path_to_subsets=path_to_subsets,
    )

    df = sweep_or_load_jobs(
        function=UnsupervisedMatching(),
        job_kwargs={
            "vision_model": vision_models,
            "language_model": language_models,
            "dataset": datasets,
            "solver": [solver],
            "distortion_metric": [distortion_metric],
            "path_to_embeddings": [path_to_embeddings],
            "num_seeds": [num_seeds],
            "seed": [seed],
            "subset": subset_fn,
            "which_seed": list(range(num_seeds)),
            "timelimit": [timelimit_sec_optimization],
        },
        num_jobs=num_jobs,
        save_path=path_to_raw_results / f"{experiment_name}.csv",
        path_to_logs=path_to_logs,
        runtime_per_function=timelimit_min_job,
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
            "slurm_nodelist": same_cpu_nodelist,
        },
    )

    # --- Result Processing and Saving ---
    # Define path for processed results
    path_to_these_results = path_to_processed_results / experiment_name
    path_to_these_results.mkdir(exist_ok=True)

    for dataset in datasets:
        df_filtered = df[df["dataset"] == str(dataset)]

        plot_accuracy_size(
            df=df_filtered,
            save_path=path_to_these_results / f"{dataset}.png",
            title=str(dataset),
        )
