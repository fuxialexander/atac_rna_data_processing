"""
Script for producing GET variant scores (motif difference * importance)
Example script for glioma and MYC gene over all cell types
"""
import os
from atac_rna_data_processing.io.mutation import CellMutCollection


# Configuration
working_dir = "/manitou/pmg/users/xf2217/interpret_natac/" # Directory with reference genome and motif file
genome_path = os.path.join(working_dir, "hg38.fa")
motif_path = os.path.join(working_dir, "NrMotifV1.pkl")

model_ckpt_path = "/manitou/pmg/projects/resources/get_interpret/pretrain_finetune_natac_fetal_adult.pth" # Model ckpt
get_config_path = "/manitou/pmg/users/xf2217/atac_rna_data_processing/atac_rna_data_processing/config/GET" # GET config

celltype_annot_path = "/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/data/cell_type_pretrain_human_bingren_shendure_apr2023.txt" # cell type mapping
celltype_dir = "/manitou/pmg/users/xf2217/Interpretation_all_hg38_allembed_v4_natac/*" # If celltype list not provided, use this directory of all cell types

celltype_path = "/manitou/pmg/users/alb2281/glioma/celltypes.txt"
variants_path = "/manitou/pmg/users/alb2281/glioma/glioma_variants.txt" # Variants of interest
genes_path = "/manitou/pmg/users/alb2281/glioma/genes.txt" # Genes of interest
normal_variants_path = "/manitou/pmg/users/xf2217/gnomad/myc.tad.vcf.gz" # gnomAD normal variants

output_dir = "/pmglocal/alb2281/repos/atac_rna_data_processing/analysis/" # Output directory
output_name = "glioma" # Experiment name
num_workers = 10 # Number of workers for parallelization


cell_mut_col = CellMutCollection(
    model_ckpt_path=model_ckpt_path,
    get_config_path=get_config_path,
    genome_path=genome_path,
    motif_path=motif_path,
    celltype_annot_path=celltype_annot_path,
    celltype_dir=celltype_dir,
    celltype_path=celltype_path,
    normal_variants_path=normal_variants_path,
    variants_path=variants_path,
    genes_path=genes_path,
    output_dir=output_dir,
    num_workers=num_workers,
)
scores = cell_mut_col.get_all_variant_scores(output_name)
