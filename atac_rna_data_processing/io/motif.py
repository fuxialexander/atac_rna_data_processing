import seqlogo
import pandas as pd
import numpy as np
from MOODS.scan import Scanner
from MOODS.parsers import pfm_to_log_odds
from MOODS.tools import threshold_from_p_with_precision

def pfm_conversion(filename, lo_bg=[2.977e-01, 2.023e-01, 2.023e-01, 2.977e-01], ps=0.01):
    mat = pfm_to_log_odds(filename, lo_bg, ps)
    if len(mat) != 4:
        return False, mat
    else:
        return True, mat


def print_results(header, seq, matrices, matrix_names, results):
    # split results into forward and reverse strands
    fr = results[:len(matrix_names)]
    rr = results[len(matrix_names):]

    # mix the forward and reverse strand results
    mixed_results = [
        [(r.pos, r.score, '+', ()) for r in fr[i]] +
        [(r.pos, r.score, '-', ()) for r in rr[i]] for i in range(len(matrix_names))]

    output = []
    for (matrix, matrix_name, result) in zip(matrices, matrix_names, mixed_results):
        # determine the length of the matrix
        l = len(matrix[0]) if len(matrix) == 4 else len(matrix[0]) + 1

        # sort the results by position
        sorted_results = sorted(result, key=lambda r: r[0])

        # create the output for each result
        for r in sorted_results:
            strand = r[2]
            pos = r[0]
            hitseq = seq[pos:pos+l]
            output.append(([header, matrix_name, pos, strand, r[1], hitseq]))
    return output


def prepare_scanner(matrices_all, bg=[2.977e-01, 2.023e-01, 2.023e-01, 2.977e-01]):
    """
    Prepare scanner for scanning motif.
    """
    bgs = {}
    thresholds = {}
    # for i in [chr]:
    #     print(i)
    #     if bg is not None:
    #         bgs[i] = bg
        # else:
            # bgs[i] = MOODS.tools.bg_from_sequence_dna(genome.get_sequence(chr), 1, 100000)
    # thresholds = {chr: [MOODS.tools.threshold_from_p_with_precision(
        # m, bgs[chr], 0.0001, 200, 4) for m in matrices_all] for chr in bgs}
    scanner = Scanner(7)
    scanner.set_motifs(matrices_all, bg,[threshold_from_p_with_precision(
        m, bg, 0.0001, 200, 4) for m in matrices_all])
    return scanner



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

    def plot_logo(self, filename=None):
        """plot seqlogo of motif using pfm file"""
        return seqlogo.seqlogo(self.pfm, filename = filename,format='png', size='small', ic_scale=True, ic_ref=0.2, ylabel='', show_xaxis=False, show_yaxis=False, show_ends=False, rotate_numbers=False, color_scheme='classic', logo_title=self.cluster_name, fineprint='')


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

    def get_gene_name_list(self):
        """Get list of gene names."""
        return np.concatenate([motif.gene_name for motif in self.motifs.values()])


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
