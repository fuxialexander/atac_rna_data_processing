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

