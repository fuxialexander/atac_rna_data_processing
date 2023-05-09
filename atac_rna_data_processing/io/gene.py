class Gene(object):
    def __init__(self, name, id, chrom, strand, tss_list) -> None:
        self.name = name
        self.id = id
        self.chrom = chrom
        self.strand = strand
        self.tss_list = tss_list
    
    def __repr__(self) -> str:
        return "Gene(name={}, id={}, chrom={}, strand={}, tss_list={})".format(self.name, self.id, self.chrom, self.strand, ','.join(self.tss_list.as_df().Start.values.astype(str)))

    @property
    def tss(self):
        # return a list of TSS objects
        return [TSS(self.name, self.id, self.chrom, self.strand, start) for start in self.tss_list.as_df().Start.values]

class TSS(object):
    def __init__(self, name, peak_id, chrom, start, strand) -> None:
        self.name = name
        self.peak_id = peak_id
        self.chrom = chrom
        self.start = start
        self.strand = strand
    
    def __repr__(self) -> str:
        return "TSS(name={}, peak_id={}, chrom={}, strand={}, start={})".format(self.name, self.peak_id, self.chrom, self.strand, str(self.start))

    def get_sample_from_peak(self, peak_df, focus=100):
        """Get the sample from the peak_df, with the peak_id as the center, and focus as the window size."""
        return peak_df.iloc[self.peak_id-focus:self.peak_id+focus]
    
class GeneExp(Gene):
    """Gene with expression data."""
    def __init__(self, name, id, chrom, strand, tss_list, exp_list) -> None:
        super().__init__(name, id, chrom, strand, tss_list)
        self.exp_list = exp_list
    
    def __repr__(self) -> str:
        return "GeneExp(name={}, id={}, chrom={}, strand={}, tss_list={}, exp={})".format(self.name, self.id, self.chrom, self.strand, ','.join(self.tss_list.as_df().Start.values.astype(str)), self.exp_list.mean())