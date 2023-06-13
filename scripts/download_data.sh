#https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/hg38.archetype_motifs.v1.0.bed.gz
#https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/hg38.archetype_motifs.v1.0.bed.gz.tbi
#https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
#https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz

#!/bin/bash

# Download multiple files into the test folder

URLs=(
    "https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/hg38.archetype_motifs.v1.0.bed.gz" 
    "https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/hg38.archetype_motifs.v1.0.bed.gz.tbi"
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz" 
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
)

# Make the test directory if it doesn't exist
mkdir -p data

for URL in "${URLs[@]}"; do
    # Get just the filename from the URL 
    FILE=${URL##*/}
    echo "Downloading $URL as $FILE..."

    # Use curl to download the file into the test folder 
    curl -L $URL -o data/$FILE  

    # Check that the file exists in the test folder and is not empty 
    if [ ! -s data/$FILE ] ; then 
        echo "Error: Failed to download $URL" 
        exit 1 
    fi 
done 

echo "All downloads completed successfully into test folder!"