import pandas as pd
import os 
from ..motif import Motif, MotifCluster, MotifClusterCollection, MotifCollection

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
        return self.annotations.Motif.unique()

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
        return self.annotations.Name.unique()
    
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