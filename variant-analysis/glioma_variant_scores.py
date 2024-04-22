"""
Script for producing GET variant scores (motif difference * importance)
GBM
"""
import os
import pandas as pd
from atac_rna_data_processing.io.mutation import CellMutCollection


def get_celltype_list(celltype_annot_path):
    celltype_annot = pd.read_csv(celltype_annot_path)
    celltype_annot_dict = celltype_annot.set_index('id').celltype.to_dict()
    id_to_celltype = celltype_annot_dict
    celltype_to_id = {v: k for k, v in id_to_celltype.items()}

    substr_list = ["Oligodendrocyte", "ligoden", "Astrocyte", "strocyte"]

    celltype_list = []
    for celltype in celltype_to_id:
        if any(substr in celltype for substr in substr_list):
            print(f"id: {celltype_to_id[celltype]}, celltype: {celltype}")
            celltype_list.append(celltype_to_id[celltype])
    return celltype_list


def get_variant_list():
    variant_list = ["rs55705857", "rs72716328", "rs147958197"]
    return variant_list


print("Starting job...")

# Configuration
celltype_annot_path = "/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/data/cell_type_pretrain_human_bingren_shendure_apr2023.txt" # list of cell types with id
celltype_list = get_celltype_list(celltype_annot_path)

# variants_path = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/input/mds_variants.csv" # map of glioma variants
variant_list = get_variant_list()

model_ckpt_path = "/manitou/pmg/projects/resources/get_interpret/pretrain_finetune_natac_fetal_adult.pth" # model ckpt
get_config_path = "/manitou/pmg/users/xf2217/atac_rna_data_processing/atac_rna_data_processing/config/GET" # GET config

working_dir = "/manitou/pmg/users/xf2217/interpret_natac/" # directory with ref genome and motif file
genome_path = os.path.join(working_dir, "hg38.fa")
motif_path = os.path.join(working_dir, "NrMotifV1.pkl")

output_name = "myc-sat-mutagen-full" # experiment name
output_dir = f"/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/{output_name}" # output directory
num_workers = 20 # number of parallel workers

print(f"Processing {len(variant_list)} variants...")

cell_mut_col = CellMutCollection(
    model_ckpt_path,
    get_config_path,
    genome_path,
    motif_path,
    celltype_annot_path,
    celltype_list,
    variant_list,
    output_dir,
    num_workers,
    run_sat_mutagen=True,
    debug=False, # if True, quick run for 5 risk variants
)
scores = cell_mut_col.get_all_variant_scores()
