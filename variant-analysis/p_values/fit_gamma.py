import pandas as pd
from scipy.stats import gamma, kstest
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def compute_p_values(score_file):
    risk_variant = os.path.basename(score_file).split(".csv")[0]
    feather_path = os.path.join(score_dir_feather, f"{risk_variant}.feather")
    results_df = pd.read_feather(feather_path).drop_duplicates()
    grouped_df = results_df.groupby(["celltype"])
    results_df_list = [] 
    
    for group, group_df in tqdm(grouped_df):
        celltype = group[0]
        score_col = group_df["score"].abs().tolist()
        score_col = [item for item in score_col if item > 0]
        score_col_norm = [item/np.max(score_col) for item in score_col]

        try:
            shape, loc, scale = gamma.fit(score_col_norm)
        except:
            print("Could not fit gamma")
            continue
        
        risk_score = group_df[group_df["variant"] == risk_variant]["score"]
        risk_score_norm = (np.abs(risk_score)).max()/np.max(score_col)

        effect_size = risk_score_norm/np.mean(score_col_norm)
        ks_statistic, ks_p_value = kstest(score_col_norm, 'gamma', args=(shape,))
        print("Kolmogorov-Smirnov Test:")
        print("KS Statistic:", ks_statistic)
        print("KS p-value:", ks_p_value)
    
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = gamma.pdf(x, shape, loc, scale)
        plt.yscale("log")
        plt.hist(score_col_norm, bins=100)
        plt.plot(x, p, 'k', linewidth=2, label='Fitted gamma')
        plt.axvline(x = risk_score_norm, color = 'r', label = 'axvline - full height')
        plt.title("variant={} celltype={}\n (n={}, effect_size={:0.3f}, KS p-value={:0.3f})".format(variant, celltype, len(score_col_norm), effect_size, ks_p_value))
        plt.xlabel("GET impact scores (norm.)")
        # index_str = index.replace("/", "")
        plt.savefig(f"{figure_dir}/{risk_variant}_{celltype}.png")
        plt.clf()
        # results_df_list.append([risk_variant, risk_score, celltype, ks_statistic, ks_p_value, effect_size])

    # results_df = pd.DataFrame(results_df_list, columns=["variant", "impact_score", "gene", "celltype", "motif", "p_value"])
    # results_df.to_csv(f"{output_dir}/{risk_variant}.csv")
    return


if __name__=="__main__":
    score_dir_csv = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/gbm-full/csv"
    score_dir_feather = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/gbm-full/feather"
    output_dir = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/gbm-full/ks_scores"
    figure_dir = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/gbm-full/fits"

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    variant = "rs55705857"
    compute_p_values(f"/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/gbm-full/feather/{variant}.csv")
