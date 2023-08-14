import pickle
import pandas as pd
try:
    from MOODS.tools import reverse_complement
except:
    print("MOODS not installed. Please install MOODS to use the reverse_complement function.")
import os 
from .motif import *

class NrMotifV1(MotifClusterCollection):
    """TFBS motif clusters defined in https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/."""
    def __init__(self, motif_dir, base_url = 
    "https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/"):
        super().__init__()
        if os.path.exists(os.path.join(motif_dir, "pfm")):
            pass
        else:
            print("Downloading PFMs...")
            os.system("cd {motif_dir} && wget --recursive --no-parent {url}pfm/".format(motif_dir=motif_dir, url=os.path.join(base_url, "pfm"))) 
        self.motif_dir = motif_dir

        self.annotations = self.get_motif_annotations(motif_dir, base_url)
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
    def load_from_pickle(cls, file_path):
        """Load the instance of the NrMotifV1 class from a pickle file."""
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        instance = cls.__new__(cls)
        instance.__setstate__(state)
        return instance

    def get_motif_annotations(self, motif_dir, base_url):
        """Get motif clusters from the non-redundant motif v1.0 release."""
        # download files
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
                    pass
                else:
                    if c in cluster_gene_list:
                        cluster_gene_list[c].append(g.upper())
                    else:
                        cluster_gene_list[c] = [g.upper()]
            if c in cluster_gene_list:
                cluster_gene_list[c] = list(set(cluster_gene_list[c]))
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



