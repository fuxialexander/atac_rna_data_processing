"""
Script for producing GET variant scores (motif difference * importance)
MDS-AML
"""
import os
import pandas as pd
from atac_rna_data_processing.io.mutation import CellMutCollection


def get_celltype_list(celltype_annot_path):
    celltype_annot = pd.read_csv(celltype_annot_path)
    celltype_annot_dict = celltype_annot.set_index('id').celltype.to_dict()
    id_to_celltype = celltype_annot_dict
    celltype_to_id = {v: k for k, v in id_to_celltype.items()}

    substr_list = ["Hematopoeitic Stem Cell"]

    celltype_list = []
    for celltype in celltype_to_id:
        if any(substr in celltype for substr in substr_list):
            print(f"id: {celltype_to_id[celltype]}, celltype: {celltype}")
            celltype_list.append(celltype_to_id[celltype])
    return celltype_list


def get_variant_to_genes_list(variants_path):
    variants_df = pd.read_csv(variants_path)
    variants_df = variants_df.drop(columns=["Unnamed: 6"]).dropna().drop_duplicates()
    variant_to_genes = variants_df.set_index('rsids')['nearest_genes'].to_dict()
    variant_to_genes = {key: [value] for key, value in variant_to_genes.items()}
    variant_list = list(variant_to_genes.keys())
    return variant_list, variant_to_genes


# Configuration
celltype_annot_path = "/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/data/cell_type_pretrain_human_bingren_shendure_apr2023.txt" # list of cell types with id
celltype_list = get_celltype_list(celltype_annot_path)

variants_path = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/input/MDS_AML_variants_genes_list.csv" # map from variant to gene of interest
variant_list, variant_to_genes = get_variant_to_genes_list(variants_path)

model_ckpt_path = "/manitou/pmg/projects/resources/get_interpret/pretrain_finetune_natac_fetal_adult.pth" # model ckpt
get_config_path = "/manitou/pmg/users/xf2217/atac_rna_data_processing/atac_rna_data_processing/config/GET" # GET config

working_dir = "/manitou/pmg/users/xf2217/interpret_natac/" # directory with ref genome and motif file
genome_path = os.path.join(working_dir, "hg38.fa")
motif_path = os.path.join(working_dir, "NrMotifV1.pkl")

output_name = "mds-aml-full" # experiment name
output_dir = f"/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/{output_name}" # output directory
num_workers = 48 # number of parallel workers

cell_mut_col = CellMutCollection(
    model_ckpt_path,
    get_config_path,
    genome_path,
    motif_path,
    celltype_annot_path,
    celltype_list,
    variant_list,
    variant_to_genes,
    output_dir,
    num_workers,
    debug=True, # quick run for 5 risk variants
)
scores = cell_mut_col.get_all_variant_scores()
