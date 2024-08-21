# Repository for data processing workflows of ATAC/RNA data
This is the repository of data processing workflows for train the regulatory models using ATAC/RNA-seq data.


## Installation

All analysis should be performed on Linux. We use Mamba to manage the environment. To install Mamba, follow instructions on https://github.com/conda-forge/miniforge#mambaforge. Choose `Mambaforge-pypy3` when installing.

After Mamba was installed, you can clone the directory and install the environment by

```{bash}
git clone git@github.com:fuxialexander/atac_rna_data_processing.git
cd atac_rna_data_processing
mamba env create -f environment.yml
<!-- or  -->
mamba install -c conda-forge -c bioconda numpy pandas pyranges scipy pyyaml zarr numcodecs pybigwig matplotlib networkx plotly seaborn tqdm cdt pysam requests seqlogo MOODS random urllib3 pyliftover hicstraw biopython gprofiler pyfaidx

pip install --user MOODS-python
```

## Usage


## Overview

The ATAC/RNA data processing workflows for one cell type involves the following main steps:

0. Confirm the genome assembly used in the data
1. Produce a BED file of the ATAC-seq peaks
2. Get motif binding instances in the ATAC-seq peaks
3. Compute a peak-by-motif binding score matrix
4. Produce a promoter expression table
5. Compute a peak expression score matrix

## 0. Confirm the genome assembly used in the data

It's crucial to make sure we are using the correct genome assembly for the data. The genome assembly used in the data can usually be found in the metadata of the data.

Once we confirmed the genome assembly we want to use, we need to specify that in the config file. This helps us download the right gene annotation files.

## 1. Produce a BED file of the ATAC-seq peaks

We want the following columns in the BED file:

1. Chromosome
2. Start position (0-based)
3. End position
4. Peak accessiblity TPM

Different dataset might store these information in differernt location, so we need to gather them and produce this file.

Here is an example of this BED file.

```{text}
chr1    0       200    0.1
chr1    1000    1320    0.2
chr1    2100    3000    3.1
```

It would be great if we can just keep using hg38/mm10. To convert a hg19/mm9 BED file to hg38/mm10, we can use the following command:

```{bash}
# Download chain file
wget http://hgdownload.cse.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz
# liftOver
liftOver -minMatch=0.5 -bedPlus=4 input.bed hg19ToHg38.over.chain.gz output.bed unlifted.bed

# Download chain file
wget http://hgdownload.cse.ucsc.edu/goldenPath/mm9/liftOver/mm9ToMm10.over.chain.gz
# liftOver
liftOver -minMatch=0.5 -bedPlus=4 input.bed mm9ToMm10.over.chain.gz output.bed unlifted.bed
```

This should keep the 4th column of the BED file intact.

## 2. Get motif binding instances in the ATAC-seq peaks

This can be achieved by running the following BASH command:

```{bash}
#Go to the root directory of the repository
export PATH=$(pwd)/scripts:$PATH
# go to data directory
cd test
get_motif.sh test hg38
# wait until the job is done
# then you can remove the intermediate files
rm -rf test.atac*chr*
```

The results are in file `test.atac.motif.bed`, which we will use in the python processing pipeline.

## 3. Compute a peak-by-motif binding score matrix

```{bash}
cd test
python test_atac.py
```

## 4. Produce a promoter expression table

## 5. Compute a peak expression score matrix

All the above steps can be done by running the following python script:

```{bash}
cd test
python test_rna.py
```
