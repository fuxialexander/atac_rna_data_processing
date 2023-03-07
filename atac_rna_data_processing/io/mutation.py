### This specify the code to
# 1. read variants from VCF files
# 2. read Structure variants from bedpe files
# 3. Manipulate the given sequence or bed files to accomodate the variants

import pandas as pd
from pysam import VariantFile
class Mutations(object):
    """Class to handle mutations
    """
    def __init__(self, vcf_file, genome):
        self.genome = genome
        self.vcf_file = vcf_file
        self.vcf = self.read_vcf()
        return
    
    def read_vcf(self):
        """Read VCF file
        """
        pd.read_csv(self.vcf_file, sep='\t', header=None)
        
        return
    

class SVs(object):
    """Class to handle SVs
    """
    def __init__(self, bedpe_file, genome):
        self.genome = genome
        self.bedpe_file = bedpe_file
        self.bedpe = self.read_bedpe()
        return
    
    def read_bedpe(self):
        """Read bedpe file
        """
        pd.read_csv(self.bedpe_file, sep='\t', header=None)
        return