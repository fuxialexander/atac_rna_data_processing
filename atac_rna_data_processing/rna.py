import os

import numpy as np
import pandas as pd

from atac_rna_data_processing.io.gencode import Gencode


def counts_to_log10tpm(counts):
    return np.log10(((counts / counts.sum(0))*1e6)+1)

def log10tpm_check(tpm):
    if (tpm.max()>6) or (tpm.min()<0):
        raise ValueError("The gene expression is not in log10(TPM+1), you should figure out the correct transformation.")

def read_rna(filename, transform=False):
    if not os.path.exists("promoter_exp.feather"):
        gencode = Gencode(assembly="hg38", version=40)
        gene_exp = pd.read_csv(filename, index_col=0)

        # if the gene expression is in log10(TPM+1), no transformation is needed
        if transform:
            gene_exp['TPM'] = counts_to_log10tpm(gene_exp.TPM.values)

        log10tpm_check(gene_exp.TPM.values)

        promoter_exp = pd.merge(gencode.gtf, gene_exp,
                                left_on='gene_name', right_index=True)
        promoter_exp.reset_index().to_feather("promoter_exp.feather")
    else:
        promoter_exp = pd.read_feather("promoter_exp.feather")
        if (promoter_exp.TPM.max()>6) or (promoter_exp.TPM.min()<0):
            os.remove("promoter_exp.feather")
            raise ValueError("The gene expression is not in log10(TPM+1), you should figure out the correct transformation.")

    return promoter_exp.drop(['level_0', 'index'], axis=1, errors='ignore')


