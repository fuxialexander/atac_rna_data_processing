class Gene(object):
    def __init__(self, name, id, chrom, strand, tss_list) -> None:
        self.name = name
        self.id = id
        self.chrom = chrom
        self.strand = strand
        self.tss_list = tss_list
    
    def __repr__(self) -> str:
        return "Gene(name={}, id={}, chrom={}, strand={}, tss_list={})".format(self.name, self.id, self.chrom, self.strand, ','.join(self.tss_list.as_df().Start.values.astype(str)))
    