## Install the conda environment using 
```
conda env create --name causal -f environment.yml
conda activate causal
```

## How to run

```bash
python -m neurips_cmi_search \
  --data-path /path/to/df_merged.parquet \
  --output-dir /path/to/output_mimic_cardiomegaly \
  --dataset-key mimic_cxr_chest \
  --gt-col gt_cardiomegaly \
  --prob-col prob_cardiomegaly \
  --subgroup-cols sex_bin,race_bin,age_bin,frontal_bin \
  --search-method both \
  --include-y-plus-searched
```
