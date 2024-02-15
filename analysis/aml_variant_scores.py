"""
Script for producing GET variant scores (motif difference * importance)
MDS-AML
"""
import os
import pandas as pd
from atac_rna_data_processing.io.mutation import CellMutCollection


def setup_files():
    celltype_annot_path = "/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/data/cell_type_pretrain_human_bingren_shendure_apr2023.txt" 
    celltype_annot = pd.read_csv(celltype_annot_path)
    celltype_annot_dict = celltype_annot.set_index('id').celltype.to_dict()
    id_to_celltype = celltype_annot_dict
    celltype_to_id = {v: k for k, v in id_to_celltype.items()}

    substr_list = ["Hematopoeitic Stem Cell"]

    celltype_id_set = set()
    for celltype in celltype_to_id:
        if any(substr in celltype for substr in substr_list):
            print(f"id: {celltype_to_id[celltype]}, celltype: {celltype}")
            celltype_id_set.add(celltype_to_id[celltype])

    with open("/pmglocal/alb2281/repos/atac_rna_data_processing/analysis/input/celltypes.txt", 'w') as file:
        for element in celltype_id_set:
            file.write(str(element) + '\n')

# setup_files()

# Configuration
working_dir = "/manitou/pmg/users/xf2217/interpret_natac/" # Directory with reference genome and motif file
genome_path = os.path.join(working_dir, "hg38.fa")
motif_path = os.path.join(working_dir, "NrMotifV1.pkl")

model_ckpt_path = "/manitou/pmg/projects/resources/get_interpret/pretrain_finetune_natac_fetal_adult.pth" # Model ckpt
get_config_path = "/manitou/pmg/users/xf2217/atac_rna_data_processing/atac_rna_data_processing/config/GET" # GET config

celltype_annot_path = "/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/data/cell_type_pretrain_human_bingren_shendure_apr2023.txt" # cell type mapping
celltype_dir = "/manitou/pmg/users/xf2217/Interpretation_all_hg38_allembed_v4_natac/*" # If celltype list not provided, use this directory of all cell types

celltype_path = "/pmglocal/alb2281/repos/atac_rna_data_processing/analysis/input/celltypes.txt"
variants_path = "/pmglocal/alb2281/repos/atac_rna_data_processing/analysis/input/MDS_AML_variants_genes_list.csv" # Genes of interest

output_dir = "/pmglocal/alb2281/repos/atac_rna_data_processing/analysis/output/aml-mds" # Output directory
output_name = "mds-aml-full" # Experiment name
num_workers = 47 # Number of workers for parallelization

cell_mut_col = CellMutCollection(
    model_ckpt_path=model_ckpt_path,
    get_config_path=get_config_path,
    genome_path=genome_path,
    motif_path=motif_path,
    celltype_annot_path=celltype_annot_path,
    celltype_dir=celltype_dir,
    celltype_path=celltype_path,
    variants_path=variants_path,
    output_dir=output_dir,
    num_workers=num_workers,
)
scores = cell_mut_col.get_all_variant_scores()