import pickle
import pandas as pd
try:
    from MOODS.tools import reverse_complement
except:
    print("MOODS not installed. Please install MOODS to use the reverse_complement function.")
import os 
from .motif import *

other_gene_mapping = {
    "ARI5B": "ARID5B",
    "HXA13": "HOXA13",
    "ATF6A": "ATF6",
    "HNF6": "ONECUT1",
    "DMRTB": "DMRTB1",
    "COE1": "EBF1",
    "EVI1": "MECOM",
    "EWSR1-FLI1": "EWSR1-FLI1",
    "ITF2": "TCF4",
    "ARNTL": "BMAL1",
    "BHE40": "BHLHE40",
    "BHLHB3": "BHLHE41",
    "BHLHB2": "BHLHE40",
    "NGN2": "NEUROG2",
    "TWST1": "TWIST1",
    "NDF2": "NEUROD2",
    "NDF1": "NEUROD1",
    "ZNF238": "ZBTB18",
    "BHA15": "BHLHA15",
    "TFE2": "TCF3",
    "ANDR": "AR",
    "HXA10": "HOXA10",
    "HXC9": "HOXC9",
    "HXA9": "HOXA9",
    "HXA1": "HOXA1",
    "HXB7": "HOXB7",
    "HXB8": "HOXB8",
    "HXC9": "HOXC9",
    "HXB13": "HOXB13",
    "HXB4": "HOXB4",
    "RAXL1": "RAX2",
    "MIX-A": "MIXL1",
    "CART1": "ALX1",
    "HEN1": "NHLH1",
    "KAISO": "ZBTB33",
    "MYBA": "MYBL1",
    "TF65": "RELA",
    "DUX": "DUX1",
    "STF1": "NR5A1",
    "ERR2": "ESRRB",
    "COT1": "NR2F1",
    "COT2": "NR2F2",
    "THA": "THRA",
    "THB": "THRB",
    "NR1A4": "RXRA",
    "RORG": "RORC",
    "ANDR": "AR",
    "PRGR": "PGR",
    "GCR": "NR3C1",
    "ERR3": "ESRRG",
    "ERR1": "ESRRA",
    "PO3F1": "POU3F1",
    "PO3F2": "POU3F2",
    "PO5F1": "POU5F1",
    "P53": "TP53",
    "P73": "TP73",
    "P63": "TP63",
    "POU5F1P1": "POU5F1B",
    "PO2F2": "POU2F2",
    "PO2F1": "POU2F1",
    "SUH": "RBPJ",
    "PEBB": "CBFB",
    "SRBP1": "SREBF1",
    "SRBP2": "SREBF2",
    "T": "TBXT",
    "BRAC": "TBXT",
    "TF2L1": "TFCP2L1",
    "TYY1": "YY1",
    "ZKSC1": "ZKSCAN1",
    "THA11": "THAP11",
    "OZF": "ZNF146",
    "ZNF306": "ZKSCAN3",
    "Z324A": "ZNF324",
    "AP2A": "TFAP2A",
    "AP2B": "TFAP2B",
    "AP2C": "TFAP2C",
    "TF7L1": "TCF7L1",
    "TF7L2": "TCF7L2",
    "STA5B": "STAT5B",
    "STA5A": "STAT5A",
    "BC11A": "BCL11A",
    "Z354A": "ZNF354A",
    "HINFP1": "HINFP",
    "PIT1": "POU1F1",
    "HTF4": "TCF12",
    "ZNF435": "ZSCAN16"
}

def fix_gene_name(x: str):
    if x.startswith('ZN') and not x.startswith('ZNF'):
        x = x.replace('ZN', 'ZNF')
    if x.startswith('ZSC') and not x.startswith('ZSCAN'):
        x = x.replace('ZSC', 'ZSCAN')
    if x.startswith('NF2L'):
        x = x.replace('NF2L', 'NFE2L')
    if x.startswith('PKNX1'):
        x = 'PKNOX1'
    if x.startswith('NKX') and '-' not in x:
        x = x[:-1] + '-' + x[-1]
    if x.startswith('PRD') and not x.startswith('PRDM'):
        x = x.replace('PRD', 'PRDM')
    if x.startswith('NFAC'):
        x = x.replace('NFAC', 'NFATC')
    if x.startswith('SMCA'):
        x = x.replace('SMCA', 'SMARCA')
    if x.startswith('ZBT') and not x.startswith('ZBTB'):
        x = x.replace('ZBT', 'ZBTB')
    # other mappings
    for k, v in other_gene_mapping.items():
        if x.startswith(k):
            x = v
    return x
    
class NrMotifV1(MotifClusterCollection):
    """TFBS motif clusters defined in https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/."""
    def __init__(self, motif_dir, base_url = 
    "https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/"):
        super().__init__()
        self.motif_dir = motif_dir
        self.annotations = self.get_motif_data(motif_dir, base_url)
        matrices = []
        matrices_rc = []
        for motif in self.get_motif_list():
            filename = os.path.join(self.motif_dir, "pfm", motif + ".pfm")
            valid=False
            if os.path.exists(filename): # let's see if it's pfm
                valid, matrix = pfm_conversion(filename)
                matrices.append(matrix)
                matrices_rc.append(reverse_complement(matrix,4))

        self.matrices = matrices
        self.matrices_all = self.matrices + matrices_rc
        self.matrix_names = self.get_motif_list()
        self.cluster_names = self.get_motifcluster_list()
        self.motif_to_cluster = self.annotations[['Motif', 'Name']].set_index('Motif').to_dict()['Name']
        self.cluster_gene_list = self.get_motifcluster_list_genes()

    # facility to export the instance as a pickle and load it back
    def __getstate__(self):
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

    def save_to_pickle(self, file_path):
        """Save the instance of the NrMotifV1 class to a pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.__getstate__(), f)

    @classmethod
    def load_from_pickle(cls, file_path, motif_dir=None):
        """Load the instance of the NrMotifV1 class from a pickle file."""
        with open(file_path, 'rb') as f:
            state = pd.read_pickle(f)
        instance = cls.__new__(cls)
        instance.__setstate__(state)
        if motif_dir is not None:
            instance.motif_dir = motif_dir
        return instance

    def get_motif_data(self, motif_dir, base_url):
        """Get motif clusters from the non-redundant motif v1.0 release."""
        # download files
        if os.path.exists(os.path.join(motif_dir, "pfm")):
            pass
        else:
            print("Downloading PFMs...")
            os.system("cd {motif_dir} && wget --recursive --no-parent {url}/".format(motif_dir=motif_dir, url=os.path.join(base_url, "pfm"))) 
        if os.path.exists(motif_dir + "motif_annotations.csv"):
            motif_annotations = pd.read_csv(motif_dir + "motif_annotations.csv")
        else:
            a = pd.read_excel(base_url + "motif_annotations.xlsx", sheet_name=1) # around 1min
            b = pd.read_excel(base_url + "motif_annotations.xlsx", sheet_name=0)
            motif_annotations = pd.merge(a, b, left_on='Cluster_ID', right_on='Cluster_ID')
            motif_annotations.to_csv("motif_annotations.csv", index=False)
        return motif_annotations

    def get_motif_list(self):
        """Get list of motifs."""
        return sorted(self.annotations.Motif.unique())

    def get_motif(self, motif_id):
        row = self.annotations[self.annotations.Motif == motif_id].iloc[0]
        return Motif(
            row.Motif, 
            row.Motif.split('_')[0].split('+'), 
            row.DBD, 
            row.Database, 
            row.Cluster_ID, 
            row.Name, 
            os.path.join(self.motif_dir, "pfm", row.Motif + ".pfm"))

    def get_motifcluster_list(self):
        """Get list of motif clusters."""
        return sorted(self.annotations.Name.unique())

    def get_motifcluster_list_genes(self):
        cluster_gene_list = {}
        for c in self.get_motifcluster_list():
            for g in self.get_motif_cluster_by_name(c).get_gene_name_list():
                if g.endswith('mouse'):
                    g = g.replace('.mouse', '').upper()
                else:
                    if c in cluster_gene_list:
                        cluster_gene_list[c].append(g.upper())
                    else:
                        cluster_gene_list[c] = [g.upper()]
            if c in cluster_gene_list:
                cluster_gene_list[c] = list(set(cluster_gene_list[c]))
        # fix the gene names in motif_gene_list
        for k in cluster_gene_list:
            cluster_gene_list[k] = [fix_gene_name(x) for x in cluster_gene_list[k]]
        return cluster_gene_list

    def get_motif_cluster_by_name(self, mc_name):
        """Get motif cluster by name."""
        mc = MotifCluster()
        mc.name = mc_name
        mc.annotations = self.annotations[self.annotations.Name == mc_name]
        mc.seed_motif = self.get_motif(mc.annotations.iloc[0].Seed_motif)
        mc.id = mc.annotations.iloc[0].Cluster_ID
        mc.motifs = MotifCollection()
        for motif_id in self.annotations[self.annotations.Name == mc_name].Motif.unique():
            mc.motifs[motif_id] = self.get_motif(motif_id)
        return mc

    def get_motif_cluster_by_id(self, mc_id):
        """Get motif cluster by id."""
        mc = MotifCluster()
        mc.name = mc_id
        mc.annotations = self.annotations[self.annotations.Cluster_ID == mc_id]
        mc.seed_motif = self.get_motif(mc.annotations.iloc[0].Seed_motif)
        mc.name = mc.annotations.iloc[0].Name

        mc.motifs = MotifCollection()
        for motif_id in self.annotations[self.annotations.Cluster_ID == mc_id].Motif.unique():
            mc.motifs[motif_id] = self.get_motif(motif_id)
        return mc

    @property
    def scanner(self, bg=[2.977e-01, 2.023e-01, 2.023e-01, 2.977e-01]):
        """Get MOODS scanner."""
        return prepare_scanner(self.matrices_all, bg)



