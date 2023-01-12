import seqlogo
import pandas as pd


class Motif(object):
    """Base class for TFBS motifs."""

    def __init__(self, id, gene_name, dbd, database, cluster_id, cluster_name, pfm):
        self.id = id
        self.gene_name = gene_name
        self.dbd = dbd
        self.database = database
        self.cluster_id = cluster_id
        self.cluster_name = cluster_name
        pfm = pd.read_csv(pfm, sep='\t', header=None).T
        pfm.columns = ['A', 'C', 'G', 'T']
        self.pfm = seqlogo.CompletePm(pfm=seqlogo.Pfm(pfm))

    def __repr__(self) -> str:
        return "Motif(id={}, gene_name={}, dbd={}, database={}, cluster_id={}, cluster_name={})".format(self.id, self.gene_name, self.dbd, self.database, self.cluster_id, self.cluster_name)

    def plot_logo(self):
        """plot seqlogo of motif using pfm file"""
        return seqlogo.seqlogo(self.pfm, format='png', size='medium')


class MotifCluster(object):
    """Base class for TFBS motif clusters."""

    def __init__(self):
        self.id = None
        self.name = None
        self.seed_motif = None
        self.motifs = MotifCollection()
        self.annotations = None

    def __repr__(self) -> str:
        return "MotifCluster(id={}, name={}, seed_motif={})".format(self.id, self.name, self.seed_motif)


class MotifClusterCollection(object):
    """List of TFBS motif clusters."""

    def __init__(self):
        super().__init__()
        self.annotations = None


class MotifCollection(dict):
    """Dictionary of TFBS motifs."""

    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return '\n'.join(self.keys())

    def get_motif_list(self):
        """Get list of motifs."""
        return list(self.keys())

    def get_motif(self, motif_id):
        """Get motif by ID."""
        return self[motif_id]
