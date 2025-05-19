<div align="center">
<h1>It's a (Blind) Match! Towards Vision–Language Correspondence without Parallel Data</h1>

[**Dominik Schnaus**](https://dominik-schnaus.github.io/)  [**Nikita Araslanov**](https://arnike.github.io/)<sup>&dagger;</sup>  [**Daniel Cremers**](https://cvg.cit.tum.de/members/cremers)<sup>&dagger;</sup>

Technical University of Munich, Munich Center of Machine Learning  <sup>&dagger;</sup>equal advising

<h3>CVPR 2025</h3>

<a href="https://arxiv.org/abs/2503.24129"><img src='https://img.shields.io/badge/ArXiv-grey' alt='Paper PDF'></a>
<a href="https://dominik-schnaus.github.io/itsamatch/"><img src='https://img.shields.io/badge/Project Page-grey' alt='Project Page URL'></a>
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<center>
  <img src="./assets/teaser.gif" width="100%">
</center>
</div>

**TL;DR:** Vision-Language models need a lot of paired training data. Can we match vision and language without any supervision? Our work shows that it could be indeed feasible.

## Abstract

The platonic representation hypothesis suggests that vision and language embeddings become more homogeneous as model and dataset sizes increase. In particular, pairwise distances within each modality become more similar. This suggests that as foundation models mature, **it may become possible to match vision and language embeddings in a fully unsupervised fashion**, i.e., without parallel data. We present the first study towards this prospect, and investigate conformity of existing vision and language foundation models in the context of "blind" matching. First, **we formulate unsupervised matching as a quadratic assignment problem** and **introduce a novel heuristic that outperforms previous solvers**. We also develop a technique to find optimal matching problems, for which a non-trivial match is very likely. Second, we conduct an **extensive study deploying a range of vision and language models on four datasets**. Our analysis reveals that for many problem instances, vision and language representations **can be indeed matched without supervision**. This finding opens possibility for exciting applications embedding semantic knowledge into other modalities. As a showcase, we demonstrate a proof-of-concept unsupervised classifier, which achieves non-trivial classification accuracy without any image-text annotation.

## News

- `19/05/2025`: Initial code released.
- `31/03/2025`: [ArXiv](https://arxiv.org/abs/2503.24129) preprint released. 🚀
- `26/02/2025`: It's a (Blind) Match! has been accepted to [CVPR](https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers)! 🎉

## Table of Contents
1. [Abstract](#abstract)
2. [News](#news)
3. [Environment Setup](#environment-setup)
4. [Reproducing Results](#reproducing-results)
    - [Configuring the paths](#configuring-the-paths)
    - [Shuffling degrades vision–language alignment (Sec. 3)](#shuffling-degrades-vision-language-alignment-sec-3)
    - [Small-scale matching (Sec. 5.1)](#small-scale-matching-sec-51)
    - [Larger-scale matching (Sec 5.2)](#larger-scale-matching-sec-52)
    - [Solver comparison (small) (Sec. 5.3)](#solver-comparison-small-sec-53)
    - [Solver comparison (larger) (Sec. 5.3)](#solver-comparison-larger-sec-53)
    - [Unsupervised classification (Sec. 5.4)](#unsupervised-classification-sec-54)
    - [Ablation on the Hahn-Grant Solver (Sec. D.3)](#ablation-on-the-hahn-grant-solver-sec-d3)
5. [Citation](#citation)

## Environment Setup

The code was tested with Python 3.12, PyTorch 2.7, and Gurobi 12.0.

1. Clone the repository:
    ```sh
    git clone https://github.com/dominik-schnaus/itsamatch.git
    ```

2. Create a conda environment:
    ```sh
    conda create -n itsamatch python=3.12
    conda activate itsamatch
    ```
  
3. Install the python packages:
    ```sh
    cd itsamatch
    pip install -r requirements.txt
    ```

4. (optional) To set up Gurobi, follow the [official guide](https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license). It is used for the larger-scale experiments and the solver comparisons.

5. Install [MPOpt](https://github.com/vislearn/libmpopt):
    ```sh
    bash install_mpopt.sh
    ```
    This installation needs to be done explicitly for the CPU that is used. If you have a different type of CPU in your cluster than locally, please run this command directly on that node.

## Reproducing Results

All experiments from the paper can be found in the folder `itsamatch/experiments`.
We use [submitit](https://github.com/facebookincubator/submitit) to run the experiments in a distributed setting.

For all experiments, first activate the `itsamatch` environment:
```sh
conda activate itsamatch
```

### Configuring the paths

In `itsamatch/experiments/utils.py`, we define 5 paths for saving the resulting embeddings, logs, subsets, pandas dataframes containing the raw results, and figures and tables from the paper with the aggregated results. Please replace the path corresponding to your file system.
```python
...
path_to_embeddings = Path("path/to/embeddings")  # Path embeddings for different models and datasets
path_to_logs = Path("path/to/logs")  # Path to store the logs from submitit
path_to_subsets = Path("path/to/subsets")  # Path to optimal subsets for larger-scale problems
path_to_raw_results = Path("path/to/raw_results")  # Path to raw experiment results as pandas dataframes
path_to_processed_results = Path("path/to/processed_results")  # Path to generated Figures and Tables
...
```

From the same document, one also needs to specify the root path for the datasets:
```python
...
data_root = "path/to/data"  # Required for all but hahn_grant_ablation.py
imagenet_root = "path/to/imagenet"  # Required for shuffle_alignment.py, larger_scale_matching.py, solver_comparison_larger.py
cococaptions_root = "path/to/cococaptions"  # Required for shuffle_alignment.py
cococaptions_json = "path/to/captions_val2017.json"  # Required for shuffle_alignment.py
...
```
Apart from ImageNet and CocoCaptions, all datasets are automatically downloaded to `data_root` when needed.

Finally, one needs to specify further information for submitit:
```python
...
cluster = "slurm"
# cluster type can be
# - "slurm": run the jobs in parallel on a slurm cluster
# - "local": run the jobs in parallel on the local machine
# - "debug": run the jobs sequentially on the local machine (also enables debugging)

# This string should specify a set of nodes that all have the same CPU to fairly compare different solvers. This string should correspond to a valid nodelist parameter from SLURM.
same_cpu_nodelist: str = None 

# This string should specify a set of nodes that can use Gurobi in parallel. This is especially useful when having an upper bound on active gurobi sessions. This string should correspond to a valid nodelist parameter from SLURM.
gurobi_nodelist: str = None
...
```
Also, one can change the mapping of colors and markers for the plots for all experiments in this file.

### Shuffling degrades vision–language alignment (Sec. 3)

This experiment shows that for all considered alignment measures, datasets, and models, the average alignment decreases strictly monotonically so that the ground-truth pairing achieves the optimal alignment. 
The experiment can be run with
```sh
python itsamatch/experiments/shuffle_alignment.py
```
It produces Fig. 2, 7, 8, and 9 from the paper.

### Small-scale matching (Sec. 5.1)

In the small-scale matching experiment, we evaluate 32 vision models and 27 language models on CIFAR-10 and CINIC-10 and observe that most models perform better than 10% accuracy. 
The experiment can be run with
```sh
python itsamatch/experiments/small_scale_matching.py
```
It results in Fig. 4, 11, 12, 13 and Tab. 3 from the paper.

### Larger-scale matching (Sec 5.2)

The larger-scale matching evaluates some models on ImageNet-100 and CIFAR-100 using optimal subsets of the classes.
We observe that all models have high accuracy for small problem sizes.
The experiment can be run with
```sh
python itsamatch/experiments/larger_scale_matching.py
```
It produces Fig. 5 from the paper.

### Solver comparison (small) (Sec. 5.3)

The solver comparison (small) evaluates different solvers on CIFAR-10 and CINIC-10. 
Our factorized Hahn-Grant solver always leads to the global optimum.
Moreover, we observe that local optima from other solvers are in general not enough to get meaningful matchings.
The experiment can be run with
```sh
python itsamatch/experiments/solver_comparison_small.py
```
It produces Tab. 1 and 7 from the paper.

### Solver comparison (larger) (Sec. 5.3)

The solver comparison (large) evaluates different solvers on CIFAR-100, showing that our factorized Hahn-Grant solver finds better solutions and tighter bounds for most problem sizes even finding the global optimum up to size 40 and outperforming commercial solvers like Gurobi.
The experiment can be run with
```sh
python itsamatch/experiments/solver_comparison_larger.py
```
It produces Fig. 6 from the paper.

### Unsupervised classification (Sec. 5.4)

In the unsupervised classification experiment, we show that fully unsupervised image classification is possible. For this, we use k-Means clustering and our matching algorithm to match cluster centers with language embeddings of class labels.
The experiment can be run with
```sh
python itsamatch/experiments/unsupervised_classification.py
```
It produces Tab. 2 and 6 from the paper.

### Ablation on the Hahn-Grant Solver (Sec. D.3)

This experiment evaluates the different design choices in our factorized Hahn-Grant solver.
It can be run with
```sh
python itsamatch/experiments/hahn_grant_ablation.py
```
It results in Tab 4. and 5 from the paper.


## Citation

If you find our work helpful, please consider citing the following paper and ⭐ the repo.

```
@inproceedings{schnaus2025it,
  title={It’s a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data},
  author={Schnaus, Dominik and Araslanov, Nikita and Cremers, Daniel},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
