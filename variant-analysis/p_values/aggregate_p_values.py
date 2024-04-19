import argparse
import os
import pandas as pd


def aggregate_p_values(args):
    p_value_dir = args.p_value_dir
    exp_name = args.exp_name

    p_value_files = os.listdir(p_value_dir)
    df_col = []

    for file in p_value_files:
        if file.endswith(".csv"):
            df = pd.read_csv(f"{p_value_dir}/{file}")
            df_col.append(df)
    
    df = pd.concat(df_col)
    df = df[df["p_value"] < 1.0]
    df = df.sort_values(by="effect_size", ascending=False)
    
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.drop("p_value_nonzero", axis=1)
    df = df.drop("effect_size_nonzero", axis=1)
    df.to_csv(f"{p_value_dir}/{exp_name}_p_values_effect_size.csv", index=False)


if __name__=="__main__":
    p_value_dir = "/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/mds-variants/p_values"

    parser = argparse.ArgumentParser(description="Compute p-values for risk variants")
    parser.add_argument("--exp_name", type=str, help="Name of experiment")
    args = parser.parse_args()

    args.p_value_dir = p_value_dir
    aggregate_p_values(args)
