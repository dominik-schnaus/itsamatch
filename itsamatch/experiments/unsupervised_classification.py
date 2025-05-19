"""
This script conducts the unsupervised classification experiments (Sec. 5.4).

It involves:
1. Extracting embeddings for various vision and language model combinations on specified datasets.
2. Computing classification accuracies for different models, datasets, and solvers.
3. Saving the results in a CSV file (Tab. 2, 6).
"""

from time import time

import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
from submitit.helpers import Checkpointable
from torch.nn.functional import normalize, one_hot
from torch.utils.data import Dataset
from tqdm import tqdm

from itsamatch.datasets import (
    CIFAR10,
    CINIC10,
)
from itsamatch.distortion_metrics import (
    CKA,
    DistortionMetric,
    GromovWasserstein,
)
from itsamatch.experiments.larger_scale_matching import (
    standardize_kernel_matrix,
)
from itsamatch.experiments.solver_comparison_small import rename_solvers
from itsamatch.experiments.utils import (
    cluster,
    data_root,
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
    LanguageModel,
    SentenceT,
)
from itsamatch.models.vision import (
    CLIP,
    DeiT,
    DINOv2,
    VisionModel,
)
from itsamatch.solvers import (
    BruteForce,
    FactorizedHahnGrant,
    MPOpt,
    OptimalTransport,
    QAPSolver,
    RandomSolver,
    ScipySolver,
)
from itsamatch.utils import (
    get_seeds,
    seed_everything,
    sweep_or_load_jobs,
)


class UnsupervisedClassification(Checkpointable):
    """
    Performs unsupervised classification by matching clustered vision embeddings
    to language embeddings using a QAP solver.
    """

    @torch.inference_mode()
    def __call__(
        self,
        vision_model: VisionModel,
        language_model: LanguageModel,
        dataset: Dataset,
        clusterer: ClusterMixin,
        solver: QAPSolver,
        distortion_metric: DistortionMetric,
        num_seeds: int = 20,
        seed: int = 42,
        timelimit: int | None = None,
        normalize_before_averaging: bool = False,
        standardize_kernel: bool = True,
    ):
        """
        Executes the unsupervised classification pipeline.

        Args:
            vision_model: The vision model to use for embeddings.
            language_model: The language model to use for embeddings.
            dataset: The dataset to use.
            clusterer: The clustering algorithm to use.
            solver: The QAP solver to use for matching.
            distortion_metric: The distortion metric to use for kernel computation.
            num_seeds: The number of random seeds for subsampling and clustering.
            seed: The global random seed.
            timelimit: Time limit for the solver in seconds.
            normalize_before_averaging: Whether to normalize embeddings before averaging (if applicable).
            standardize_kernel: Whether to standardize kernel matrices.

        Returns:
            A pandas DataFrame containing the results of the experiment.
        """
        seeds = get_seeds(seed, num_seeds)

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

        num_labels = language_embeddings.shape[0]
        clusterer.n_clusters = num_labels

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

        # iterate over different random seeds for robustness
        for local_seed in tqdm(seeds):
            seed_everything(local_seed)

            # subsample vision embeddings and labels
            sampled_indices = torch.randperm(
                num_samples,
                generator=torch.Generator().manual_seed(local_seed),
            )[: num_samples // 2]
            vision_embeddings_subsampled = vision_embeddings[sampled_indices]
            labels_subsampled = labels[sampled_indices]

            # cluster vision embeddings
            clusterer.fit(vision_embeddings_subsampled.cpu().numpy())
            clustered_vision_embeddings = torch.tensor(
                clusterer.cluster_centers_
            ).to(vision_embeddings)
            cluster_assignments = torch.tensor(clusterer.labels_).to(
                vision_embeddings.device
            )

            # normalize clustered vision embeddings
            clustered_vision_embeddings = normalize(
                clustered_vision_embeddings, dim=-1
            )

            # compute vision kernel matrix
            vision_kernel_matrix = distortion_metric.kernel_source(
                clustered_vision_embeddings
            )

            if standardize_kernel:
                # standardize kernel matrix
                vision_kernel_matrix = standardize_kernel_matrix(
                    vision_kernel_matrix
                )

            start_time = time()

            # solve the QAP problem
            if solver.factorized:
                cost1 = -distortion_metric.loss.h1(vision_kernel_matrix)
                cost2 = distortion_metric.loss.h2(language_kernel_matrix)
                constant = (
                    distortion_metric.loss.f1(vision_kernel_matrix).sum()
                    + distortion_metric.loss.f2(language_kernel_matrix).sum()
                )
                permutation, optimal, cost, bound = solver.solve(
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
                permutation, optimal, cost, bound = solver.solve(
                    full_cost_matrix, timelimit=timelimit
                )

            end_time = time()
            time_elapsed = end_time - start_time

            # compute the accuracy of the predicted matching
            prediction = permutation.to(cluster_assignments)[
                cluster_assignments
            ]
            accuracy = (prediction == labels_subsampled).float().mean().item()

            # compute the best possible accuracy given the clustering (oracle matching)
            frequencies = one_hot(
                cluster_assignments.long(), num_labels
            ).T @ one_hot(labels_subsampled, num_labels)
            _, col_ind = linear_sum_assignment(-frequencies)
            ground_truth_matching = torch.as_tensor(
                col_ind, device=cluster_assignments.device
            )
            ground_truth_accuracy = (
                (
                    ground_truth_matching[cluster_assignments]
                    == labels_subsampled
                )
                .float()
                .mean()
                .item()
            )

            # compute the matching accuracy (how well the solver's permutation matches the oracle permutation)
            matching_accuracy = (
                (permutation == ground_truth_matching).float().mean().item()
            )

            # collect results for this seed
            results_dict = {
                "Accuracy": accuracy,
                "Ground truth accuracy": ground_truth_accuracy,
                "Matching accuracy": matching_accuracy,
                "local_seed": local_seed,
                "prediction": prediction.tolist(),
                "permutation": permutation.tolist(),
                "Optimal": optimal,
                "Cost": cost,
                "Bound": bound,
                "Time elapsed": time_elapsed,
            }
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
        df["clusterer"] = repr(clusterer)
        df["solver"] = str(solver)
        df["timelimit"] = timelimit
        df["seed"] = seed
        return df


if __name__ == "__main__":
    # --- Experiment Configuration ---
    experiment_name = "unsupervised_classification"

    num_jobs = 30

    vision_models = [
        CLIP(name="ViT-L/14@336"),
        DeiT(name="DeiT-B/16d@384"),
        DINOv2(name="ViT-G/14"),
    ]

    language_models = [
        SentenceT(name="All-Roberta-large-v1"),
        SentenceT(name="all-mpnet-base-v2"),
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

    solver = FactorizedHahnGrant()
    distortion_metric = GromovWasserstein()

    num_seeds = 20

    clusterer = KMeans(
        n_clusters=None,
        init="k-means++",
        n_init=100,
        random_state=seed,
    )

    recompute = False

    timelimit_min_job = 6 * 60  # 6 hours

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

    # --- Matching (Model Comparison) ---
    # Run the unsupervised classification for different models and datasets
    df_models = sweep_or_load_jobs(
        function=UnsupervisedClassification(),
        job_kwargs={
            "vision_model": vision_models,
            "language_model": language_models,
            "dataset": datasets,
            "clusterer": [clusterer],
            "solver": [solver],
            "distortion_metric": [distortion_metric],
            "num_seeds": [num_seeds],
            "seed": [seed],
        },
        num_jobs=num_jobs,
        save_path=path_to_raw_results / f"{experiment_name}_models.csv",
        path_to_logs=path_to_logs,
        runtime_per_function=timelimit_min_job,
        cluster=cluster,
        recompute=recompute,
        condition=None,
        shuffle=True,
        max_num_timeout=0,
        submitit_kwargs={
            "name": f"{experiment_name}_models",
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

    # Accuracy for different vision/language models and datasets (Tab. 2):
    mean = (
        (
            df_models.pivot_table(
                values="Accuracy",
                index=["dataset", "vision_model_class"],
                columns="language_model_name",
                aggfunc="mean",
            )
            * 100
        )
        .round(1)
        .astype(str)
    )

    std = (
        (
            df_models.pivot_table(
                values="Accuracy",
                index=["dataset", "vision_model_class"],
                columns="language_model_name",
                aggfunc="std",
            )
            * 100
        )
        .round(1)
        .astype(str)
    )
    dataset_order = [str(dataset) for dataset in datasets]
    vision_model_order = [
        vision_model.__class__.__name__ for vision_model in vision_models
    ]
    language_model_order = [
        language_model.name for language_model in language_models
    ]
    out = (mean + " ± " + std).loc[
        (dataset_order, vision_model_order), language_model_order
    ]
    out.to_csv(path_to_these_results / "tab_2.csv")

    # --- Matching Experiment (Solver Comparison) ---
    # Define solvers to compare
    solvers = [
        BruteForce(),
        OptimalTransport(),
        ScipySolver(method="faq", num_repetitions=1),
        ScipySolver(method="faq", num_repetitions=1, transpose_first=True),
        MPOpt(),
        FactorizedHahnGrant(),
        RandomSolver(),
    ]

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

    # Run the unsupervised classification for different solvers
    df_solvers = sweep_or_load_jobs(
        function=UnsupervisedClassification(),
        job_kwargs={
            "vision_model": vision_models,
            "language_model": language_models,
            "dataset": datasets,
            "clusterer": [clusterer],
            "solver": solvers,
            "distortion_metric": get_distortion_metrics,
            "num_seeds": [num_seeds],
            "seed": [seed],
        },
        num_jobs=num_jobs,
        save_path=path_to_raw_results / f"{experiment_name}_solvers.csv",
        path_to_logs=path_to_logs,
        runtime_per_function=timelimit_min_job,
        cluster=cluster,
        recompute=False,
        condition=None,
        shuffle=True,
        submitit_kwargs={
            "name": f"{experiment_name}_solvers",
            "nodes": 1,
            "tasks_per_node": 1,
            "cpus_per_task": 4,
            "gpus_per_node": 0,
            "mem_gb": 64,
        },
    )

    # --- Result Processing and Saving (Solver Comparison) ---
    # Accuracy and Matching Accuracy for different solvers (Tab. 6)
    df_solvers = rename_solvers(df_solvers)

    # Add Ground Truth (GT) results by using the ground truth accuracy
    df_solvers_gt = df_solvers[df_solvers["solver"] == "Ours"]
    df_solvers_gt.loc[:, "Accuracy"] = df_solvers_gt["Ground truth accuracy"]
    df_solvers_gt.loc[:, "solver"] = "GT"
    df_solvers = pd.concat([df_solvers, df_solvers_gt], ignore_index=True)

    mean = (
        df_solvers.pivot_table(
            values=["Accuracy", "Matching accuracy"],
            index=["dataset", "vision_model_class", "solver"],
            columns="language_model_name",
            aggfunc="mean",
        )
        .round(2)
        .astype(str)
        .swaplevel(0, 1, axis=1)
    )

    std = (
        df_solvers.pivot_table(
            values=["Accuracy", "Matching accuracy"],
            index=["dataset", "vision_model_class", "solver"],
            columns="language_model_name",
            aggfunc="std",
        )
        .round(2)
        .astype(str)
        .swaplevel(0, 1, axis=1)
    )

    solver_order = [
        "Random",
        "LocalCKA",
        "OT",
        "FAQ",
        "MPOpt",
        "Ours",
        "GT",
    ]
    dataset_order = [str(dataset) for dataset in datasets]
    vision_model_order = [
        vision_model.__class__.__name__ for vision_model in vision_models
    ]
    language_model_order = [
        language_model.__class__.__name__ for language_model in language_models
    ]
    out = (mean + " ± " + std).loc[
        (dataset_order, vision_model_order, solver_order), language_model_order
    ]
    out.to_csv(path_to_these_results / "tab_6.csv")
