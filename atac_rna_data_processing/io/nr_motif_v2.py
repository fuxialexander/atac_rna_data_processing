import os
import pandas as pd
import pandas as pd
try:
    from MOODS.tools import reverse_complement
except:
    print("MOODS not installed. Please install MOODS to use the reverse_complement function.")
import os 
from .motif import *
import numpy as np

class NrMotifV2(MotifClusterCollection):
    """
    TFBS motif clusters defined in the V2 release.
    """
    def __init__(self, motif_dir, base_url="https://resources.altius.org/~jvierstra/projects/motif-clustering-v2.1beta/"):
        super().__init__()
        self.motif_dir = motif_dir
        self.annotations = self.get_motif_data(motif_dir, base_url)
        self.matrices, self.matrices_all, self.matrix_names = self.process_motif_files()

    def get_motif_data(self, motif_dir, base_url):
        """
        Get motif clusters from the non-redundant motif v2.0 release.
        """
        annotations_file = os.path.join(motif_dir, "metadata.tsv")
        if os.path.exists(annotations_file):
            motif_annotations = pd.read_csv(annotations_file, sep="\t")
        else:
            print("Annotations file not found. Please ensure it's downloaded.")
            # Code to download or process annotations can be added here.
        return motif_annotations

    def process_motif_files(self):
        """
        Process motif files to extract matrices and other relevant data.
        """
        meme_file = os.path.join(self.motif_dir, "consensus_pwms.meme")
        motifs = parse_meme_file(meme_file)

        matrices = motifs  # This assumes that `parse_meme_file` returns an array of matrices.
        matrices_all = np.concatenate([matrices, matrices[:, ::-1, ::-1]], axis=0)  # Assuming reverse complement is needed.
        matrix_names = [motif["name"] for motif in motifs]

        return matrices, matrices_all, matrix_names

    # Other methods from NrMotifV1 can be adapted here with modifications as needed.

def parse_meme_file(file_path):
    """
    Parse the .meme file in the V2 format to extract motif matrices.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    motifs = []
    current_motif = None
    lines = lines[9:]  # Skipping header lines specific to the file format
    for line in lines:
        if line.startswith("MOTIF"):
            if current_motif is not None:
                motifs.append(current_motif)
            current_motif = {"name": line.split()[1], "letter_prob_matrix": []}
        elif "letter-probability matrix" in line:
            letter_prob_matrix = []
            width = int(line.split("w=")[1].split()[0])
            for j in range(width):
                idx = lines.index(line) + j + 1
                row = list(map(float, lines[idx].strip().split()))
                letter_prob_matrix.append(row)
            current_motif["letter_prob_matrix"] = np.array(letter_prob_matrix)
            motifs.append(current_motif)
            current_motif = None  # Reset for the next motif

    # Process motifs to standardize or apply any specific modifications needed
    # This part can be customized based on requirements

    return motifs  # Adjusted to return a list of motif dictionaries

# Additional class definitions, such as MotifCluster, Motif, etc., should be adapted to match the V2 data structures.
