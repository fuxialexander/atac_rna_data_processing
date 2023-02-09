from pyranges import read_gtf
from pyranges import PyRanges as pr
import pandas as pd
import os
from ..io.gene import Gene
from ..io.gencode import Gencode
from ..io.atac import *

class REData(object):
    """Reads ATAC-seq and RNA-seq data from a given sample
    """
    def __init__(self, sample, assembly="hg38", gencode_version=40):
        super(REData, self).__init__()
        self.sample = sample
        self.assembly = assembly
        self.gencode_version = gencode_version
        self.gencode = Gencode(assembly=self.assembly, version=self.version)
        self.atac = ATAC(proj=self.proj)
        self.rna = RNA(proj=self.proj)
        return



def main(sample, peak_motif_bed, peak_motif_csv, output_dir):
    atac, bed = read_atac(sample)
    rna = read_multiome_rna(sample)
    peak_motifs = get_peak_motif(sample, peak_motif_bed, bed, peak_motif_csv)
    for label, content in atac.iteritems():
        celltype = label[5:]
        print(celltype)
        if not path.exists(path.join(output_dir, sample + "." + celltype.replace('.', '_') + ".watac.npz")):
            nonzero_indices = np.where(content.values > 0)[0]
            peak_motif = peak_motifs.iloc[nonzero_indices].dropna()
            celltype_data = get_celltype_data(content, nonzero_indices, peak_motif)
            celltype_data.iloc[:, 0:3].to_csv(path.join(output_dir, sample + "_" + celltype.replace('.', '_') + ".csv"))
            save_npz(path.join(output_dir, sample + "_" + celltype.replace('.', '_') + ".watac.npz"),csr_matrix(celltype_data.iloc[:,3:286].values))
            celltype_data['Accessibility'] = 1
            save_npz(path.join(output_dir,sample + "_" + celltype.replace('.', '_') + ".natac.npz"),csr_matrix(celltype_data.iloc[:,3:286].values))
        if not path.exists(path.join(output_dir, sample + "_" + celltype.replace('.', '_') + ".exp.npy")):
            sample_bed = pr(bed.reset_index().iloc[nonzero_indices, :])
            sample_exp = sample_bed.join(pr(rna).extend(100), how='left')
            sample_exp = sample_exp.as_df()[['index', 'Strand', 'RNA.'+celltype]].groupby(['index', 'Strand']).mean()
            sample_exp = sample_exp.reset_index()
            sample_exp_neg = sample_exp[sample_exp.Strand=="-"].iloc[:,2].fillna(0)
            sample_exp_neg[sample_exp_neg<0] = 0
            sample_exp_pos = sample_exp[sample_exp.Strand=="+"].iloc[:,2].fillna(0)
            sample_exp_pos[sample_exp_pos<0] = 0
            sample_exp_neg_tss = (sample_exp[sample_exp.Strand=="-"].iloc[:,2] >= 0).fillna(False)
            sample_exp_pos_tss = (sample_exp[sample_exp.Strand=="+"].iloc[:,2] >= 0).fillna(False)
            np.save(path.join(output_dir, sample + "_" + celltype.replace('.', '_') + ".tss.npy"), np.stack([sample_exp_pos_tss,sample_exp_neg_tss]).T)
            np.save(path.join(output_dir, sample + "_" + celltype.replace('.', '_') + ".exp.npy"), np.stack([sample_exp_pos,sample_exp_neg]).T) 
    return