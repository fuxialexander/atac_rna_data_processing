import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm


def compute_p_value_from_scores(score_file):
    """
    Computes p-value from a score file for a single variant
    Score file contains scores for risk variant (risk_variant.feather) and nearby normal variants
    """

    risk_variant = os.path.basename(score_file).split(".csv")[0]
    feather_path = os.path.join(score_dir_feather, f"{risk_variant}.feather")
    results_df = pd.read_feather(feather_path).drop_duplicates()
    grouped_df = results_df.groupby(["gene", "celltype", "index"])  
    results_df_list = []

    print(f"Starting {risk_variant}...")
    
    for group, group_df in tqdm(grouped_df):
        gene, celltype, index = group
        risk_score = group_df[group_df["variant"] == risk_variant]["score"].values
        assert(len(risk_score)==1)
        risk_score = risk_score[0]

        background_scores = group_df["score"].values
        p_value = 1.0 - np.sum(risk_score > background_scores)/len(background_scores)
        results_df_list.append([risk_variant, risk_score, gene, celltype, index, p_value])

    results_df = pd.DataFrame(results_df_list, columns=["variant", "impact_score", "gene", "celltype", "motif", "p_value"])
    results_df.to_csv(f"/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/gbm-full/p_values_final/{risk_variant}.csv")
    return


if __name__=="__main__":
    score_dir_csv = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/gbm-full/csv"
    score_dir_feather = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/gbm-full/feather"
    output_dir = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/gbm-full/p_values_final"

    parser = argparse.ArgumentParser(description="Compute p-values for risk variants")
    parser.add_argument("--file_path", type=str, help="Path to score file")
    args = parser.parse_args()

    compute_p_value_from_scores(args.file_path)
