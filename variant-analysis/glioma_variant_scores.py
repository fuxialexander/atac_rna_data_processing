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


def get_variant_to_genes_list(variants_path, genes_list):
    # variants_df = pd.read_csv(variants_path, sep="\t")
    # variant_list = list(set(list(variants_df.ID.values)))
    variant_list = ['rs55705857', 'rs72716328', 'rs147958197']
    variant_to_genes = {rsid: genes_list for rsid in variant_list}
    return variant_list, variant_to_genes


print("Starting job...")

# Configuration
celltype_annot_path = "/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/data/cell_type_pretrain_human_bingren_shendure_apr2023.txt" # list of cell types with id
celltype_list = get_celltype_list(celltype_annot_path)

variants_path = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/input/glioma_variants.txt" # map of glioma variants
genes_list = ['ABCA7', 'ACBD6', 'AFAP1L1', 'AFAP1L2', 'AMPD3', 'ANKRD33B', 'ANKRD49', 'ANKRD7', 'ANO3', 'ARHGAP22', 'ARL6IP4', 'ARMCX5', 'ARPC1B', 'ASIC4', 'ASPHD1', 'ATP2A1', 'BAIAP2L1', 'BCYRN1', 'BEAN1', 'BLOC1S4', 'BLOC1S5-TXNDC5', 'BVES-AS1', 'C2orf74', 'C6orf226', 'CALCB', 'CAV1', 'CAV2', 'CBR3', 'CCDC26', 'CDK18', 'CLDN11', 'CMTM7', 'CMTM8', 'COL19A1', 'CPED1', 'CPM', 'CRHR1', 'CRYAB', 'CTHRC1', 'DCP1A', 'DGCR5', 'DLGAP1-AS4', 'DNAJC28', 'EDAR', 'EHD4', 'ELOVL7', 'ENPP1', 'EPS8L2', 'EPSTI1', 'ETS2', 'EXOC3-AS1', 'FA2H', 'FALEC', 'FAM156A', 'FAM184B', 'FERMT3', 'FKBP5', 'FMN1', 'FMO5', 'GADD45G', 'GALNT12', 'GATC', 'GDNF-AS1', 'GGACT', 'GLDN', 'GLRA3', 'GLS2', 'GMFG', 'GPR17', 'GPR89B', 'GS1-124K5.4', 'HAP1', 'HAS2-AS1', 'HERC5', 'HLA-F', 'HLF', 'HMX1', 'HOXD-AS2', 'HOXD3', 'ITGA4', 'IZUMO4', 'KANSL1-AS1', 'KCNIP2-AS1', 'KCNS3', 'KMT2B', 'KMT2E-AS1', 'LACTB2-AS1', 'LCP1', 'LIN28B', 'LINC00310', 'LINC00886', 'LINC01036', 'LINC01116', 'LINC01117', 'LINC01170', 'LINC01285', 'LONRF3', 'LRP4-AS1', 'LRRK1', 'LURAP1L-AS1', 'LYPD5', 'MAPT-AS1', 'MCF2L2', 'MIR210HG', 'MIR34AHG', 'MMP28', 'MNX1', 'MR1', 'MRPS21', 'MT1M', 'MTHFD2', 'MX1', 'MYRF', 'MYT1', 'NABP1', 'NBPF15', 'NCOA4', 'NFE2L3', 'NPR3', 'NSUN5', 'NUTM2B-AS1', 'ONECUT1', 'OSGEPL1-AS1', 'OTUD7B', 'P4HA2', 'P4HA3', 'PAXIP1-AS2', 'PCOLCE2', 'PDE6B', 'PDP2', 'PET117', 'PIGZ', 'PLA2R1', 'PLLP', 'PLS1', 'PLSCR2', 'POLR3G', 'POM121C', 'POMZP3', 'PPM1M', 'PPP1R3B', 'PRH1', 'PRKG2', 'PSMB9', 'RAB26', 'RARRES1', 'RBAK-RBAKDN', 'RBMS3-AS3', 'RN7SL832P', 'RNF115', 'RRP8', 'SAMD9', 'SCAMP1-AS1', 'SCOC-AS1', 'SEC22B', 'SHOX2', 'SLC16A3', 'SLC1A1', 'SLC25A53', 'SLC38A11', 'SLC46A3', 'SLC9A2', 'SLC9A5', 'SMIM10L1', 'SMIM18', 'SP2-AS1', 'SPON1', 'SRGAP2', 'ST7-AS1', 'STAP2', 'STC2', 'STEAP2', 'STX3', 'TAF15', 'TBC1D8B', 'TCAF2', 'TFEB', 'TFR2', 'TH2LCRR', 'TMC6', 'TMCO4', 'TMEM177', 'TMEM185A', 'TMEM44-AS1', 'TPD52L1', 'TRAM2-AS1', 'TRIM14', 'TSPAN10', 'TWIST1', 'TXNIP', 'UCN', 'UGT8', 'UNC13D', 'URB1-AS1', 'USF1', 'USHBP1', 'VAMP8', 'VAX2', 'VSTM5', 'VWDE', 'XKR9', 'ZCCHC3', 'ZDHHC11B', 'ZDHHC23', 'ZNF213', 'ZNF257', 'ZNF316', 'ZNF385B', 'ZNF778', 'ZNF98', 'ZNF99']
variant_list, variant_to_genes = get_variant_to_genes_list(variants_path, genes_list)

model_ckpt_path = "/manitou/pmg/projects/resources/get_interpret/pretrain_finetune_natac_fetal_adult.pth" # model ckpt
get_config_path = "/manitou/pmg/users/xf2217/atac_rna_data_processing/atac_rna_data_processing/config/GET" # GET config

working_dir = "/manitou/pmg/users/xf2217/interpret_natac/" # directory with ref genome and motif file
genome_path = os.path.join(working_dir, "hg38.fa")
motif_path = os.path.join(working_dir, "NrMotifV1.pkl")

output_name = "myc-continuous" # experiment name
output_dir = f"/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/{output_name}" # output directory
num_workers = 10 # number of parallel workers

print(f"Processing {len(variant_list)} variants...")

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
    debug=False, # if True, quick run for 5 risk variants
)
scores = cell_mut_col.get_all_variant_scores()
