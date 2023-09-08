# An lentiMPRA experiment is defined by multiple input
# 1. Cell types (as in celltype.py) which as corresponding ATAC-seq (and RNA-seq) data
# 2. Sequences of elements to test
# 3. Reporter vector and function to insert the elements
# 4. A specific model checkpoint to use for the in silico predictions

# there are two possibilities of the cell type: 
# 1. the cell type has been involved in the training or evaluation of the model, i.e. the cell type is in-distribution
# 2. the cell type is out-of-distribution, and need finetuning or scaling to make the model work

# This file defines several useful classes to handle the above input, 
# 1. it uses GETCellType from celltype.py to handle ATAC-seq and RNA-seq data
# 2. it uses DNASequence and DNASequenceCollection to handle sequences. A user definable function is used to insert the given list of sequence-of-interest into the reporter vector sequence
# 3. it defines a ObsesrvedLentiMPRA class to handle the sequence-of-interest (Location: [Chromosome, Start, End], and DNA Seuqence), reporter vector, and measured outcome as well as other metadata  for each of the sequence-of-interest.
# 4. it defines a InSilicoLentiMPRA class to specify the cell type, the model checkpoints, and the sequence-of-interest. It also specifies the function to generate the in silico prediction for the sequence-of-interest.
# 5. Each InsilicoLentiMPRA can host a expression prediction checkpoint and optionally a chromatin accessibility prediction checkpoint. If the later is avaliable, the InsilicoLentiMPRA can also generate the chromatin accessibility prediction for the sequence-of-interest. and the eventual prediction of MPRA activity will be a product of the expression and chromatin accessibility prediction.

# The InSilicoLentiMPRA has a load_data function to load precomputed in silico prediction for the sequence-of-interest. The inputs includes
