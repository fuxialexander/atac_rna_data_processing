#%%
import os

from pyranges import PyRanges as pr
import pyBigWig as bw
#%%
class BigWigOverRegion:
    def __init__(self, bigwig_file_path, regions: pr):
        self.bigwig_file_path = bigwig_file_path
        self.bw = bw.open(self.bigwig_file_path)
        self.regions = regions
        self.average_values = self.get_average_values(self.regions)

    def get_average_values(self, regions):
        # Compute the average value over each region in the given regions
        # using pyBigWig and pyranges
        # Return a list of average values
        def compute_average(region):
            chrom = region.Chromosome[0]
            start = region.Start[0]
            end = region.End[0]
            value = self.bw.stats(chrom, start, end)
            if value[0] is None:
                return 0
            else:
                return value[0]

        values = regions.apply(compute_average)
        return values
    
class CageBigWigOverRegion(BigWigOverRegion):
    def __init__(self, plus_bigwig_file_path, minus_bigwig_file_path, regions: pr):
        self.plus_bigwig_file_path = plus_bigwig_file_path
        self.minus_bigwig_file_path = minus_bigwig_file_path
        self.regions = regions
        self.plus_bw = bw.open(self.plus_bigwig_file_path)
        self.minus_bw = bw.open(self.minus_bigwig_file_path)
        self.average_values = self.get_average_values(self.regions)

    def get_average_values(self, regions):
        # Compute the average value over each region in the given regions
        # using pyBigWig and pyranges
        # Annotate the regions with gencode such that it's possible to query the expression values of each promoters
        # Return a list of average values
        def compute_average(region):
            chrom = region.Chromosome[0]
            start = region.Start[0]
            end = region.End[0]
            plus_value = self.plus_bw.stats(chrom, start, end)
            minus_value = self.minus_bw.stats(chrom, start, end)
            if plus_value[0] is None:
                plus_value = 0
            else:
                plus_value = plus_value[0]
            if minus_value[0] is None:
                minus_value = 0
            else:
                minus_value = minus_value[0]
            return (plus_value, minus_value)

        values = regions.apply_chunks(compute_average)
        return values
# %%
from pyranges import PyRanges as pr, read_bed
atac = read_bed("../../human/k562/k562_cut0.03.atac.bed")
test = BigWigOverRegion("../../human/k562/ENCFF754EAC.bigWig", atac)
# %%
