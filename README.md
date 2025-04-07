<div align="center">
<h1>It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data</h1>

[**Dominik Schnaus**](https://dominik-schnaus.github.io/)  [**Nikita Araslanov**](https://arnike.github.io/)<sup>&dagger;</sup>  [**Daniel Cremers**](https://cvg.cit.tum.de/members/cremers)<sup>&dagger;</sup>

Technical University of Munich, Munich Center of Machine Learning  <sup>&dagger;</sup>equal advising

<h3>CVPR 2025 Highlight</h3>

<a href="https://arxiv.org/abs/2503.24129"><img src='https://img.shields.io/badge/ArXiv-grey' alt='Paper PDF'></a>
<a href="https://dominik-schnaus.github.io/itsamatch/"><img src='https://img.shields.io/badge/Project Page-grey' alt='Project Page URL'></a>
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<center>
  <img src="./assets/teaser.gif" width="100%">
</center>
</div>

**⚠️We are currently cleaning up the code and will upload it in the coming weeks.⚠️**

**TL;DR:** Vision-Language models need a lot of paired training data. Can we match vision and language without any supervision? Our work shows that it could be indeed feasible.

## Abstract

The platonic representation hypothesis suggests that vision and language embeddings become more homogeneous as model and dataset sizes increase. In particular, pairwise distances within each modality become more similar. This suggests that as foundation models mature, **it may become possible to match vision and language embeddings in a fully unsupervised fashion**, i.e., without parallel data. We present the first study towards this prospect, and investigate conformity of existing vision and language foundation models in the context of "blind" matching. First, **we formulate unsupervised matching as a quadratic assignment problem** and **introduce a novel heuristic that outperforms previous solvers**. We also develop a technique to find optimal matching problems, for which a non-trivial match is very likely. Second, we conduct an **extensive study deploying a range of vision and language models on four datasets**. Our analysis reveals that for many problem instances, vision and language representations **can be indeed matched without supervision**. This finding opens possibility for exciting applications embedding semantic knowledge into other modalities. As a showcase, we demonstrate a proof-of-concept unsupervised classifier, which achieves non-trivial classification accuracy without any image-text annotation.

## News

- `31/03/2025`: [ArXiv](https://arxiv.org/abs/2503.24129) preprint released. 🚀
- `26/02/2025`: It's a (Blind) Match! has been accepted to [CVPR](https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers)! 🎉

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