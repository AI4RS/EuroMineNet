# EuroMineNet
<div align="center">

### EuroMineNet: A Multitemporal Sentinel-2 Benchmark for Spatiotemporal Mining Footprint Analysis in the European Union (2015–2024)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.isprsjprs.2026.04.046-blue)](https://doi.org/10.1016/j.isprsjprs.2026.04.046)
[![Dataset DOI](https://rodare.hzdr.de/badge/DOI/10.14278/rodare.4656.svg)](https://doi.org/10.14278/rodare.4656)

Published in [ISPRS Journal of Photogrammetry and Remote Sensing 2026](https://www.sciencedirect.com/science/article/pii/S092427162600225X)  

**Weikang Yu**<sup>1,2</sup>, **Vincent Nwazelibe**<sup>2</sup>, **Xianping Ma**<sup>3</sup>, **Xiaokang Zhang**<sup>4</sup>, **Richard Gloaguen**<sup>2</sup>, **Xiao Xiang Zhu**<sup>1</sup>, **Pedram Ghamisi**<sup>2,5</sup>

<sup>1</sup> Technical University of Munich, <sup>2</sup> Helmholtz-Zentrum Dresden-Rossendorf (HZDR),  <sup>3</sup> Southwest Jiaotong University, <sup>4</sup> Wuhan University, <sup>5</sup> Lancaster University
</div>




## Overview

EuroMineNet is a multitemporal benchmark designed for spatiotemporal mining footprint analysis using Sentinel-2 imagery across mining sites in the European Union. This repository provides the official code and utilities associated with the EuroMineNet paper, including dataset loading, preprocessing, postprocessing, and split-generation scripts.

## Paper

The paper is available online:

- ScienceDirect: https://www.sciencedirect.com/science/article/pii/S092427162600225X
- DOI: https://doi.org/10.1016/j.isprsjprs.2026.04.046

## Dataset

The EuroMineNet dataset is publicly available on RODARE:

- Dataset record: https://rodare.hzdr.de/record/4656
- Dataset DOI: https://doi.org/10.14278/rodare.4656

The dataset release contains the archive:

```text
EuroMineNet.zip
```

Please download the dataset from RODARE and follow the repository instructions below to organize the files before running the scripts.

## Repository Structure

```text
EuroMineNet/
├── preprocess.py
├── postprocess.py
├── datasets/
│   ├── cd_dataset.py
│   ├── seg_dataset.py
│   └── train_val_test_splits.py
├── LICENSE
└── README.md
```

### Main Components

- `preprocess.py`  
  Data preparation and preprocessing entry point.

- `postprocess.py`  
  Postprocessing utilities used after model prediction.

- `datasets/cd_dataset.py`  
  Dataset utilities for change detection experiments.

- `datasets/seg_dataset.py`  
  Dataset utilities for segmentation and mining footprint mapping experiments.

- `datasets/train_val_test_splits.py`  
  Utilities for generating or loading train/validation/test splits.

## Quick Start

### 1. Download the dataset

Download the dataset from RODARE:

```text
https://rodare.hzdr.de/record/4656
```

After downloading, extract the archive:

```bash
unzip EuroMineNet.zip -d data/
```

A typical local data organization may look like:

```text
data/
└── EuroMineNet/
    ├── Site1
        ├── images/
        └── labels/
    ...
    └── SiteN
        ├── images/
        └── labels/
```

Please adjust the paths in the scripts according to your local data directory.

### 2. Run preprocessing for cropping

```bash
python preprocess.py
```

### 3. Run postprocessing for reconstruction

```bash
python postprocess.py
```

## Citation

If you use the EuroMineNet dataset or code in your research, please cite the paper and the dataset.

```bibtex
@article{yu2026eurominenet,
  title   = {EuroMineNet: A multitemporal Sentinel-2 benchmark for spatiotemporal mining footprint analysis in the European Union (2015--2024)},
  author  = {Yu, Weikang and Nwazelibe, Vincent and Ma, Xianping and Zhang, Xiaokang and Gloaguen, Richard and Zhu, Xiao Xiang and Ghamisi, Pedram},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  year    = {2026},
  doi     = {10.1016/j.isprsjprs.2026.04.046}
}
```

or
```text
Yu, W., Nwazelibe, V., Ma, X., Zhang, X., Gloaguen, R., Zhu, X.X. and Ghamisi, P. (2026) ‘EuroMineNet: A multitemporal Sentinel-2 benchmark for spatiotemporal mining footprint analysis in the European Union (2015–2024)’, ISPRS Journal of Photogrammetry and Remote Sensing, 237, pp. 409–425. doi: 10.1016/j.isprsjprs.2026.04.046.
```

## License

The EuroMineNet dataset is released under the **Creative Commons Attribution 4.0 International License** according to the RODARE dataset record.

## Contact

For questions, issues, or suggestions related to this repository, please open an issue on GitHub.

For questions related to the paper, please refer to the correspondence information provided in the published article.

## Acknowledgements

We thank all contributors and institutions involved in the development of EuroMineNet. Please refer to the paper for the full acknowledgements and funding information.