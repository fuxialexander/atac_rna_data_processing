import pandas as pd
import numpy as np
import os


def compute_p_value_from_scores(score_file):
    """
    Computes p-value from a score file for a single variant
    Score file contains scores for risk variant (risk_variant.csv) and nearby normal variants
    """
    risk_variant = os.path.basename(score_file).split(".csv")[0]
    results_df = pd.read_csv(score_file)
    motif_list = results_df["index"].unique()

    results_df_list = []        
    for motif in motif_list:
        motif_df = results_df[results_df["index"] == motif]
        risk_score = motif_df[motif_df["variant"] == risk_variant]["score"].values[0]
        gene = motif_df[motif_df["variant"] == risk_variant]["gene"].values[0]
        background_scores = motif_df["score"].values
        p_value = 1.0 - np.sum(risk_score > background_scores)/len(background_scores) # one-tailed test
        results_df_list.append([risk_variant, motif, p_value, gene])

    results_df = pd.DataFrame(results_df_list, columns=["variant", "motif", "p-value", "gene"])
    return results_df


score_dir = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/mds-aml-full/csv"
output_dir = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/mds-aml-full/p_values"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

p_value_df_col = []
for file in os.listdir(score_dir):
    p_value_df = compute_p_value_from_scores(os.path.join(score_dir, file))
    p_value_df_col.append(p_value_df)

p_value_df = pd.concat(p_value_df_col, ignore_index=True)
p_value_df.to_csv(f"{output_dir}/p_values.csv")
