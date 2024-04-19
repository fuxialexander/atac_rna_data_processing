#!/bin/bash

file_dir=/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/mds-variants/csv
log_dir=/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/mds-variants/logs
for entry in "$file_dir"/*
do
    python /pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/p_values/compute_p_values.py --file_path "$entry" > $log_dir/$(basename "$entry").log 2>&1 &
done

wait
echo "Done with all jobs"
