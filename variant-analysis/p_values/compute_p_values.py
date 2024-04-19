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
    results_df = pd.read_csv(score_file)
    if "Unnamed: 0" in results_df.columns:
        results_df = results_df.drop(columns=["Unnamed: 0"])
    results_df = results_df.drop_duplicates()
    results_df["score_abs"] = np.abs(results_df["score"])
    grouped_df = results_df.groupby(["gene", "celltype", "index"])  
    results_df_list = []

    print(f"Starting {risk_variant}...")
    
    for group, group_df in tqdm(grouped_df):
        gene, celltype, index = group
        risk_score = group_df[group_df["variant"] == risk_variant]["score_abs"].values
        assert(len(risk_score)==1)
        risk_score = risk_score[0]

        background_scores = group_df["score_abs"].values
        background_nonzero = [item for item in background_scores if item > 0]

        p_value = 1.0 - np.sum(risk_score > background_scores)/len(background_scores)
        p_value_nonzero = 1.0 - np.sum(risk_score > background_nonzero)/len(background_nonzero)

        effect_size = risk_score/np.mean(background_scores)
        effect_size_nonzero = risk_score/np.mean(background_nonzero)
        results_df_list.append([risk_variant, risk_score, gene, celltype, index, p_value, p_value_nonzero, effect_size, effect_size_nonzero])

    results_df = pd.DataFrame(results_df_list, columns=["variant", "impact_score", "gene", "celltype", "motif", "p_value", "p_value_nonzero", "effect_size", "effect_size_nonzero"])
    results_df.to_csv(f"{output_dir}/{risk_variant}.csv")
    return



if __name__=="__main__":
    output_dir = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/mds-variants/p_values"

    parser = argparse.ArgumentParser(description="Compute p-values for risk variants")
    parser.add_argument("--file_path", type=str, help="Path to score file")
    args = parser.parse_args()

    compute_p_value_from_scores(args.file_path)
