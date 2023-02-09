# %%
import sys

sys.path.append('..')
from atac_rna_data_processing.lib.GENA_LM.src.gena_lm.modeling_bert import \
    BertForPreTraining
from atac_rna_data_processing.io.region import *
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch
import hicstraw
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
model = BertForPreTraining.from_pretrained('AIRI-Institute/gena-lm-bert-base')
model.eval()
model.to('cuda')

# %%
def bert_inference(model, tokenizer, seq):
    input_ids = tokenizer(seq)['input_ids']
    max_len = np.array([len(input_id) for input_id in input_ids]).max()
    input_ids = [input_id + [3] * (max_len - len(input_id)) # 3 is the id of [PAD]
                 for input_id in input_ids]
    input_ids = torch.tensor(input_ids)
    sequence_output, _ = model(input_ids.cuda().to(torch.int32))
    return sequence_output[:, 0, :] # [CLS] token embedding

def bert_inference_batch(model, tokenizer, seqs, batch_size=100):
    output = []
    for i in tqdm(range(len(seqs)//batch_size)):
        with autocast(dtype=torch.float16):
            output.append(bert_inference(model, tokenizer, [
                     s.seq for s in seqs[i*batch_size:(i+1)*batch_size]]).detach().cpu().numpy())
    return np.concatenate(output, axis=0)
# %%
# hg19 = Genome('hg19', '/home/xf2217/Projects/common/hg19.fasta')
hg38 = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')
hic_hg38 = hicstraw.HiCFile("/home/xf2217/Projects/geneformer_nat/data/H1_ESC.hic")
# %%
i=0
for chr in list(hg38.chrom_sizes.keys())[:24]:
    tiles = hg38.tiling_region(chr, 4000000, 2000000)
    for tile in tqdm(tiles):
        seqs = tile.tiling_region(200, 200).collect_sequence(upstream=20, downstream=20)
        embedding = bert_inference_batch(model, tokenizer, seqs, batch_size=100)
        
        np.save(f'../data/genome/hg38/{chr}_{tile.start}_{tile.end}.npy', embedding)
        tile_hic = tile.get_hic(hic_hg38, resolution=25000)
        np.save(f'../data/genome/hg38/{chr}_{tile.start}_{tile.end}_hic.npy', tile_hic)

# %%
