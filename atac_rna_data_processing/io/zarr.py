import zarr
import torch
import numpy as np
from get_model.model import MotifScanner
from torch.cuda.amp import autocast

class ZarrGenome:
    """
    Represents a genome stored in a Zarr file format.

    This class provides functionality to interact with a genome sequence data stored in Zarr format.
    It supports operations such as indexing, slicing, and obtaining motif scores for specific regions
    in a chromosome.

    Attributes:
        zarr_file_path (str): Path to the Zarr file containing genome data.
        model (MotifScanner): Pre-trained model to calculate motif scores.

    Methods:
        __getitem__(chr, key): Returns a slice of the genome sequence for a given chromosome.
        get_motif_scores(chr, regions): Returns motif scores for specified regions in a chromosome.
    """

    def __init__(self, zarr_file_path, model_path):
        """
        Initializes the ZarrGenome with the given Zarr file and a trained model.

        Args:
            zarr_file_path (str): Path to the Zarr file containing genome data.
            model_path (str): Path to the trained model for motif scoring.
        """
        self.zarr_file_path = zarr_file_path
        self.genome_data = zarr.load(zarr_file_path)
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Loads the MotifScanner model from the specified path.

        Args:
            model_path (str): Path to the trained model.

        Returns:
            MotifScanner: Loaded model.
        """
        model = MotifScanner()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.requires_grad_(False)
        model.cuda()
        return torch.compile(model)

    def __getitem__(self, chr, key):
        """
        Returns a slice of the genome sequence for a given chromosome.

        Args:
            chr (str): Chromosome identifier (e.g., 'chr1', 'chr2', ...).
            key (slice): A slice object defining the start and end positions.

        Returns:
            ndarray: Slice of the genome sequence.
        """
        return self.genome_data[chr][key]

    def get_motif_scores(self, chr, regions):
        """
        Returns motif scores for specified regions in a chromosome.

        Args:
            chr (str): Chromosome identifier.
            regions (list of tuples): List of (start, end) tuples defining regions.

        Returns:
            ndarray: Motif scores for the specified regions.
        """
        results = []
        with autocast():
            for start, end in regions:
                sequence = torch.Tensor(self.genome_data[chr][0][start:end]).unsqueeze(0).cuda()
                sequence.requires_grad_(False)
                score = self.model(sequence)
                results.append(score.cpu().detach().numpy())
        return np.array(results)
