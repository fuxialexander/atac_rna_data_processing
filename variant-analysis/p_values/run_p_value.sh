#!/bin/bash

file_dir=/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/gbm-full/csv
log_dir=/pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/output/gbm-full/p_values_logs
for entry in "$file_dir"/*
do
    python /pmglocal/alb2281/repos/atac_rna_data_processing/variant-analysis/compute_risk_p_values_v2.py --file_path "$entry" > $log_dir/$(basename "$entry").log 2>&1 &
done

wait
echo "Done with all jobs"
