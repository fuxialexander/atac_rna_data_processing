# %%
import sys
sys.path.append('..')
from atac_rna_data_processing.rna import RNA
# %%
a = RNA(sample='test', assembly='hg38', version=40, transform=True, tf_list="../human/tf_list.csv")
# %%
a.export_data()
# %%
a.get_tss_atac_idx('chr1', 778769)
# %%
a.get_gene('RET')
# %%
a.rna
# %%
from pyranges import read_bed
atac = read_bed("test.atac.bed")
# %%
atac
# %%
from pyranges import PyRanges as pr
b  = atac.join(pr(a.rna), how = 'outer')
# %%
import seaborn as sns

covered_genes = b[b.Start!=-1].as_df().gene_name.unique()
uncovered_genes = b[b.Start==-1].as_df().groupby('gene_name').TPM.mean()
uncovered_genes = uncovered_genes[~uncovered_genes.index.isin(covered_genes)]
sns.displot(uncovered_genes)
# %%
