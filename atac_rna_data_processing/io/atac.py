from os import path
import numpy as np
import pandas as pd
from pyranges import read_bed
from pyranges import PyRanges as pr
from scipy.sparse import csr_matrix, save_npz
import yaml

from atac_rna_data_processing.io.gencode import Gencode

def read_bed4(bedfile, filtered=True):
    """
    Reads a 4-columns BED file. Rename the columns to Chromosome, Start, End, and TPM
    """
    if filtered:
        bed = pr(read_bed(bedfile, as_df=True).rename({'Name':'Accessibility'}, axis=1).query('Accessibility>0'), int64=True)
    else:
        bed = pr(read_bed(bedfile, as_df=True).rename({'Name':'Accessibility'}, axis=1), int64=True)
    return bed

class ATAC(object):
    """class of ATAC-seq data
    Use the __init__() function to assign values to object properties, or other operations that are necessary to do when the object is being created
The self parameter is a reference to the current instance of the class, and is used to access variables that belongs to the class.
    """

    def __init__(self, sample, assembly, version=40, scanned_motif=False, union_motif=False, tf_list=None):
        super(ATAC, self).__init__()
        self.sample = sample
        self.assembly = assembly
        self.version = version

        if path.exists(self.sample + ".atac.motif.output.feather"):
            return self.load_from_feather(self.sample + ".atac.motif.output.feather", tf_list)

        if union_motif:
            peak_bed = pr(self.read_atac().as_df().reset_index(), int64=True)
            motif_df = peak_bed.join(union_motif,nb_cpu=28).as_df().pivot_table(index='index', columns='Cluster', values='Score', aggfunc='sum').fillna(0).reset_index()
            motif_df = pd.merge(peak_bed.as_df(), motif_df, left_on='index', right_on='index').drop('index', axis=1)
            motif_df.to_feather(self.sample + ".atac.motif.output.feather")
            return self.load_from_feather(self.sample + ".atac.motif.output.feather", tf_list)
            
        self.peak_bed = self.read_atac()
        self.accessibility = self.peak_bed.as_df().iloc[:,3].values
        self.promoter_atac = self.get_promoter_atac()
        self.tf_atac = self.get_tf_atac(tf_list)
        if scanned_motif:
            self.peak_motif_feather = self.get_peak_motif()
            self.motif_data = self.get_motif_data()
            self.motif_dict = {motif: i for i, motif in enumerate(
                self.motif_data.columns[3:])}
            if not path.exists(self.sample + ".csv"):
                self.export_data()
        return
#Returns a string as a representation of the object.
    def __repr__(self) -> str:
        repr_str =  f"ATAC-seq data of {self.sample}\nAssembly: {self.assembly}\nNumber of ATAC peaks: {self.accessibility.shape[0]}"
        if hasattr(self, 'motif_data'):
            repr_str += f"\nNumber of motifs: {self.motif_data.shape[0]}"
        return repr_str


    def load_from_feather(self, feather_file, tf_list):
        self.peak_motif_feather = feather_file
        motif_feather = pd.read_feather(feather_file)
        motif_feather.iloc[:, 4:] = motif_feather.iloc[:, 4:]/motif_feather.iloc[:, 4:].max()
        self.motif_data = motif_feather
        self.peak_bed = pr(motif_feather[['Chromosome', 'Start', 'End']], int64=True)
        self.accessibility = self.peak_bed.as_df().iloc[:,3].values
        self.promoter_atac = self.get_promoter_atac()
        self.tf_atac = self.get_tf_atac(tf_list)
        self.motif_dict = {motif: i for i, motif in enumerate(
            self.motif_data.columns[3:])}
        if not path.exists(self.sample + ".csv"):
                self.export_data()
        return
        

#Reads a atac from a file. Return a csr_array with TPM
    def read_atac(self):
        """
        Reads a atac from a file. Return a csr_array with TPM
        """
        bed = read_bed4(self.sample + ".atac.bed")
        # bed = atac.reset_index()['index'].str.split("-", expand=True)
        # bed.columns = ['Chromosome', 'Start', 'End']
        # bed.Start = bed.Start.astype(int)
        # bed.End = bed.End.astype(int)
        return bed  # [nonzero_indices, :], nonzero_indices
    
#df means dataframe, which is a 2-D labeled data structure with columns of potentially different types
    def get_motif_cutoff(tf, peak_motif):
        df = peak_motif[tf].values
        df = df[(df > 0)]
        ## why do we divide by 10 the length of the dataframe?
        new_len = len(df)//10
        #argsort returns the indices that would sort an array
        df = df[df.argsort()][-new_len]
        return df
        
    #Reads a peak motif bed file and returns a dataframe with the motifs in each peaks
    def get_peak_motif(self, with_cutoff=False):
        """
        Reads a peak motif bed file and returns a dataframe with the motifs in each peaks

        Args:
        peak_motif_bed (str): path to the peak motif bed file
        output_file (str): path to the output file
        """
        if path.exists(self.sample + ".atac.motif.output.feather"):
            return self.sample + ".atac.motif.output.feather"
        else:
            peaks = self.peak_bed.as_df().reset_index()
            peak_motif = pd.read_csv(self.sample + ".peak_motif.bed", sep='\t', header=None, names=[
                                     'chr', 'start', 'end', 'motif', 'score'])
            peak_motif = peak_motif.pivot(
                index=['chr', 'start', 'end'], columns='motif', values='score').reset_index().fillna(0)
            peak_motif = pd.merge(peaks, peak_motif, left_on=[
                                  'Chromosome', 'Start', 'End'], right_on=['chr', 'start', 'end'], how='left')
            peak_motif.drop(['chr', 'start', 'end'], axis=1, inplace=True)
            peak_motif.set_index('index', inplace=True)
            if with_cutoff:
                cutoff = [self.get_motif_cutoff(
                    i, peak_motif) for i in peak_motif.columns[3:]]
            else:
                cutoff = [0 for i in peak_motif.columns[3:]]
            peak_motif.iloc[:, 3:] = (peak_motif.iloc[:, 3:].values > np.array(
                cutoff))*1 * peak_motif.iloc[:, 3:].values
            peak_motif.reset_index(drop=True).to_feather(
                self.sample + ".atac.motif.output.feather")
        return self.sample + ".atac.motif.output.feather"

    def normalize(self, x):
        return (x-x.min())/(x.max()-x.min())

    # def get_tf_accessibility(self, tf_list):
    #     if tf_list == None:
    #         return None
    #     else:
    #         tf_list = pd.read_csv(tf_list, header=0).gene_name.values
    #         gencode = Gencode(assembly=self.assembly, version=self.version)
    #         gencode_peak = self.peak_bed.join(pr(gencode.gtf, int64=True).extend(100), how='left').as_df()
    #         return gencode_peak.query('gene_name in @tf_list').groupby('gene_name').Score.mean().reindex(tf_list, fill_value=0)

    def get_motif_data(self, final_index=None):
        """
        Reads a data atac and a peak motif dataframe and returns a dataframe with the motifs and the accessibility

        Args:
        atac (csr_array): atac with TPM
        peak_motif (dataframe): dataframe with the peak motifs
        """
        peak_motif = pd.read_feather(self.peak_motif_feather)
        if final_index:
            index_ac = np.where(
                np.isin(final_index, peak_motif.index.values.astype(int)))[0]
            index_pm = np.where(
                np.isin(peak_motif.index.values.astype(int), final_index))[0]
        else:
            index_ac = np.arange(self.accessibility.shape[0])
            index_pm = np.arange(peak_motif.shape[0])
        celltype_data = peak_motif.iloc[index_pm].fillna(0)
        # accessibility = np.log2(self.accessibility[index_ac]+1)
        celltype_data['Accessibility'] = accessibility  # / accessibility.max()
        celltype_data.iloc[:, 4:-1] = self.normalize(celltype_data.iloc[:, 4:-1].values) #/ celltype_data.iloc[:, 4:-1].values.max()
        return celltype_data

    def get_promoter_atac(self, force=False):
        """Read the gene expression data."""
        if not path.exists(self.sample + ".promoter_atac.feather") or force:
            gencode = Gencode(assembly=self.assembly, version=self.version)
            # gene_exp = pd.read_csv(self.sample + ".rna.csv", index_col=0)

            # if the gene expression is in log10(TPM+1), no transformation is needed
            # if self.transform:
                # gene_exp['TPM'] = counts_to_log10tpm(gene_exp.TPM.values)

            # log10tpm_check(gene_exp.TPM.values)

            promoter_atac = (pr(gencode.gtf, int64=True)
                            .extend(100)
                            .join(self.peak_bed, how='left')
                            .as_df())
            promoter_atac.Accessibility.replace(-1, 0, inplace=True)
            promoter_atac = promoter_atac.groupby(
                ['gene_name', 'gene_id']).Accessibility.mean().reset_index()
            promoter_atac.to_feather(self.sample + ".promoter_atac.feather")
        else:
            promoter_atac = pd.read_feather(
                self.sample + ".promoter_atac.feather")
            # if (promoter_exp.TPM.max() > 6) or (promoter_exp.TPM.min() < 0):
            #     os.remove(self.sample + "promoter_exp.feather")
            #     raise ValueError(
            #         "The gene expression is not in log10(TPM+1), you should figure out the correct transformation.")

        return promoter_atac#.drop(['level_0', 'index'], axis=1, errors='ignore')

    def get_tf_atac(self, tf_list):
        """Get the ATAC data of transcription factors."""
        if tf_list is None:
            return None
        else:
            tf_list = pd.read_csv(tf_list, header=0).gene_name.values
            tf_atac = self.promoter_atac.query("gene_name in @tf_list")
            return tf_atac.groupby('gene_name').Accessibility.mean().reindex(tf_list, fill_value=0)

    def export_data(self):
        """
        Exports the data to a YAML file, a csv file and a npz file,
        """
        metadata = {'sample': self.sample,
                    'assembly': self.assembly,
                    # 'motif_dict': self.motif_dict,
                    'peak_motif_feather': self.peak_motif_feather}
        # save metadata to YAML file named self.sample + ".yaml"
        with open(self.sample + ".yml", 'w') as outfile:
            yaml.dump(metadata, outfile, default_flow_style=False)
        self.motif_data.iloc[:, 0:3].to_csv(self.sample + ".csv")
        save_npz(self.sample + ".watac.npz",
                 csr_matrix(self.motif_data.iloc[:, 3:].values))
        tmp_motif_data = self.motif_data.copy()
        tmp_motif_data['Accessibility'] = 1
        save_npz(self.sample + ".natac.npz",
                 csr_matrix(tmp_motif_data.iloc[:, 3:].values))

        if self.tf_atac is not None:
            np.save(self.sample + ".tf_atac.npy", self.tf_atac.Accessibility.values)



class ATACWithSequence(ATAC):
    
    pass


