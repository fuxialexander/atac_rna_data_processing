#!/bin/bash
#SBATCH --account=pmg
#SBATCH --job-name=causal
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/pmglocal/xf2217/slurm.%N.%j.out
#SBATCH --error=/manitou/pmg/users/xf2217/tmp/slurm.%N.%j.err
#SBATCH --exclude=m001,m010,m013,m012
mkdir -p /pmglocal/xf2217/tmp
export TMPDIR=/pmglocal/xf2217/tmp/
source /manitou-home/home/xf2217/.bashrc
mamba activate /manitou-home/home/xf2217/.conda/envs/tsne
mkdir -p /pmglocal/xf2217/tmp/TFAtlas/
mkdir -p /pmglocal/xf2217/tmp/Interpretation_all_hg38_allembed_v4_natac
# test if /pmglocal/xf2217/tmp/TFAtlas/$1 exist before copy
if [ -d "/pmglocal/xf2217/tmp/TFAtlas/$1" ]; then
    echo "Directory /pmglocal/xf2217/tmp/TFAtlas/$1 exists."
else
    cp -r /manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/TFAtlas/$1.* /pmglocal/xf2217/tmp/TFAtlas/
fi
# test if /pmglocal/xf2217/tmp/Interpretation_all_hg38_allembed_v4_natac/$1 exist before copy
if [ -d "/pmglocal/xf2217/tmp/Interpretation_all_hg38_allembed_v4_natac/$1" ]; then
    echo "Directory /pmglocal/xf2217/tmp/Interpretation_all_hg38_allembed_v4_natac/$1 exists."
else
    cp -r /manitou/pmg/users/xf2217/Interpretation_all_hg38_allembed_v4_natac/$1 /pmglocal/xf2217/tmp/Interpretation_all_hg38_allembed_v4_natac/
fi

python /manitou/pmg/users/xf2217/atac_rna_data_processing/test/test_causal.py $1

cp -r /pmglocal/xf2217/tmp/Interpretation_all_hg38_allembed_v4_natac/$1/allgenes/$1.zarr/causal* /manitou/pmg/users/xf2217/Interpretation_all_hg38_allembed_v4_natac/$1/allgenes/$1.zarr/
rm -rf /pmglocal/xf2217/tmp/TFAtlas/$1.*
rm -rf /pmglocal/xf2217/tmp/Interpretation_all_hg38_allembed_v4_natac/$1

