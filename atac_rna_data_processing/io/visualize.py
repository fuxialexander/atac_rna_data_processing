from pysam import tabix_compress, tabix_index
import pandas as pd

class TrackHub(object):
    """A class for creating a UCSC track hub from a list of BedGraph files."""
    def __init__(self, base_dir, hub_name) -> None:
        self.base_dir = base_dir
        self.hub_name = hub_name

def write_dataframe_to_bedgraph(df, output_file):
    # Make sure df has Chromosome, Start, End, and Score columns
    # Write DataFrame to BedGraph file
    df.to_csv(output_file, sep='\t', header=False, index=False)

    # Compress using bgzip
    compressed_file = f"{output_file}.gz"
    pysam.tabix_compress(output_file, compressed_file)

    # Index using tabix
    pysam.tabix_index(compressed_file, preset="bed")

# Example usage:
# Assuming you have a DataFrame named 'data' and an output file named 'output.bedgraph'
write_dataframe_to_bedgraph(data, 'output.bedgraph')
