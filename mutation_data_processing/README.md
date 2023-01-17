# *in-silico* mutagenesis pipeline

## Installation

All analysis should be performed on Linux. We use Mamba to manage the environment. To install Mamba, follow instructions on https://github.com/conda-forge/miniforge#mambaforge. Choose `Mambaforge-pypy3` when installing.

After Mamba was installed, you can clone the directory and install the environment by

```{bash}
git clone 
cd 
mamba env create -f environment.yml
```


## Usage

## Mutation data processing and prediction

This is the repository of data processing and analysis workflows for *in-silico* mutagenesis using regulatory models.

The following workflows are included:

```{text}
├── human
│   ├── 10x (hg38)
│   ├── aGBM (?)
│   ├── fetal_human (hg19)
│   ├── k562 (hg38)
│   ├── pGBM (?)
│   └── TCGA (hg19)
├── mouse
```

## Overview
