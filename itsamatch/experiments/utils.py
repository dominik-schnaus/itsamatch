"""Utility variables and configurations for experiments."""

from pathlib import Path

from matplotlib import pyplot as plt

from itsamatch.utils import seed_everything

# Define paths for storing experiment-related data.
# These paths need to be configured based on the user's environment.
path_to_embeddings = Path("path/to/embeddings")  # Path embeddings for different models and datasets
path_to_logs = Path("path/to/logs")  # Path to store the logs from submitit
path_to_subsets = Path("path/to/subsets")  # Path to optimal subsets for larger-scale problems
path_to_raw_results = Path("path/to/raw_results")  # Path to raw experiment results as pandas dataframes
path_to_processed_results = Path("path/to/processed_results")  # Path to generated Figures and Tables

# Define root paths for datasets.
# These paths are required for specific experiments.
data_root = "path/to/data"  # Required for all but hahn_grant_ablation.py
imagenet_root = "path/to/imagenet"  # Required for shuffle_alignment.py, larger_scale_matching.py, solver_comparison_larger.py
cococaptions_root = "path/to/cococaptions"  # Required for shuffle_alignment.py
cococaptions_json = "path/to/captions_val2017.json"  # Required for shuffle_alignment.py

# Set a global seed for reproducibility.
seed = 42
seed_everything(seed)

# Cluster configuration.
# These settings need to be adjusted based on the execution environment.
cluster = "slurm"
# cluster type can be
# - "slurm": run the jobs in parallel on a slurm cluster
# - "local": run the jobs in parallel on the local machine
# - "debug": run the jobs sequentially on the local machine (also enables debugging)

# This string should specify a set of nodes that all have the same CPU to fairly compare different solvers. This string should correspond to a valid nodelist parameter from SLURM.
same_cpu_nodelist: str = None 

# This string should specify a set of nodes that can use Gurobi in parallel. This is especially useful when having an upper bound on active gurobi sessions. This string should correspond to a valid nodelist parameter from SLURM.
gurobi_nodelist: str = None

# Define a color palette for plotting.
# Used to maintain consistent colors across different visualizations.
palette = {
    "CLIP": "#E37222",
    "DeiT": "#A2AD00",
    "DINOv2": "#0065BD",
    "Random ViT": "#98C6EA",
    "Pixel values": "#A02B93",
    "Random": "#000000",
    "ConvNeXt": "#98C6EA",
    "DINO": "#A02B93",
    "LocalCKA": "#A02B93",
    "OT": "#E37222",
    "FAQ": "#A2AD00",
    "MPOpt": "#98C6EA",
    "Gurobi": "#FED802",
    "FactorizedHahnGrant": "#0065BD",
    "Ours": "#0065BD",
}

# Define markers for plotting.
# Used to maintain consistent marker styles across different visualizations.
markers = {
    "CLIP": "s",
    "DeiT": "o",
    "DINOv2": "p",
    "Random ViT": "^",
    "Pixel values": "d",
    "Random": ",",
    "ConvNeXt": "^",
    "DINO": "d",
    "LocalCKA": "p",
    "OT": "*",
    "FAQ": "^",
    "MPOpt": "d",
    "Gurobi": "o",
    "FactorizedHahnGrant": "s",
    "Ours": "s",
}

# Configure font settings for plots.
plt.rcParams.update(
    {
        "font.size": 22,
    }
)
plt.rc("axes", unicode_minus=False)

# Define additional plot settings.
markersize = 10
dash = (3.7, 1.6)
