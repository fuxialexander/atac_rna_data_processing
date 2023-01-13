#!/bin/bash
# this script takes the atac peaks for a specific chromosome, intersects it with the motif peaks, sorts it based on chromosome, start, end and motif, groups it by chromosome, start, end and motif and sums the counts, sorts it again and saves the result to a file named ${SAMPLE}.atac.peak_motif_$1

# $1 and $2 are the variables passed to the script, $1 is assumed to be the chromosome and $2 is assumed to be the sample name

# the script uses bedtools and sort commands to perform its operations.



# Sample prefix name

SAMPLE=$2

# Intersect the atac peaks with the motif peaks to get the peak_motifs

bedtools intersect -a ${SAMPLE}.atac.$1 -b ${SAMPLE}.atac.motif.$1 -wa -wb | cut -f1,2,3,7,8,10 > ${SAMPLE}.atac.peak_motif_$1

# sort the peak_motifs based on chromosome, start, end, and motif

sort -k1,1 -k2,2 -k3,3 -k4,4 ${SAMPLE}.atac.peak_motif_$1 -o ${SAMPLE}.atac.peak_motif_$1

# group by chromosome, start, end, and motif and sum the counts

bedtools groupby -i ${SAMPLE}.atac.peak_motif_$1 -g 1-4 -c 5 -o sum > ${SAMPLE}.atac.peak_motif_$1_tmp;

# move the temporary file to the original file name

mv ${SAMPLE}.atac.peak_motif_$1_tmp ${SAMPLE}.atac.peak_motif_$1;

# sort the peak_motifs based on chromosome, start, end and motif

sort -k1,1V -k2,2n -k3,3n -k4,4 ${SAMPLE}.atac.peak_motif_$1 -o ${SAMPLE}.atac.peak_motif_$1