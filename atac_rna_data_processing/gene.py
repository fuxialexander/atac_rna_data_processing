class Gene(object):
    def __init__(self, name, id, chrom, strand, tss_list) -> None:
        self.name = name
        self.id = id
        self.chrom = chrom
        self.strand = strand
        self.tss_list = tss_list
    
    def __repr__(self) -> str:
        return "Gene(name={}, id={}, chrom={}, strand={}, tss_list={})".format(self.name, self.id, self.chrom, self.strand, ','.join(self.tss_list.as_df().Start.values.astype(str)))
    
class GeneExp(Gene):
    """Gene with expression data."""
    def __init__(self, name, id, chrom, strand, tss_list, exp_list) -> None:
        super().__init__(name, id, chrom, strand, tss_list)
        self.exp_list = exp_list
    
    def __repr__(self) -> str:
        return "GeneExp(name={}, id={}, chrom={}, strand={}, tss_list={}, exp={})".format(self.name, self.id, self.chrom, self.strand, ','.join(self.tss_list.as_df().Start.values.astype(str)), self.exp_list.mean())