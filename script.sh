
module load miniconda
module load python3/3.8

export PYTHONPATH=$PWD

python -m causal_analysis \
  --data-path results/cardiomegaly/mimic_cxr/full_finetune/resnet50/seed_42/predictions.csv\
  --output-dir /output/mimic/cardiomegaly/resnet50_FT \
  --dataset-key mimic_cxr_chest \
  --split-col split \
  --train-split train \
  --val-split validate \
  --test-split test \
  --gt-col gt_cardiomegaly \
  --prob-col prob_cardiomegaly \
  --subgroup-cols sex_bin,race_bin,age_bin,frontal_bin \
  --search-method both \
  --include-y-plus-searched


python -m causal_analysis \
  --data-path results/pleural_effusion/mimic_cxr/full_finetune/resnet50/seed_42/predictions.csv\
  --output-dir /output/mimic/pleural_effusion/resnet50_FT \
  --dataset-key mimic_cxr_chest \
  --split-col split \
  --train-split train \
  --val-split validate \
  --test-split test \
  --gt-col gt_pleural_effusion \
  --prob-col prob_pleural_effusion \
  --subgroup-cols sex_bin,race_bin,age_bin,frontal_bin \
  --search-method both \
  --include-y-plus-searched


python -m causal_analysis \
  --data-path results/cardiomegaly/mimic_cxr/full_finetune/resnet50/seed_42/predictions.csv \
  --output-dir /output_debug_exact/mimic/cardiomegaly/resnet50_FT \
  --dataset-key mimic_cxr_chest \
  --gt-col gt_cardiomegaly \
  --prob-col prob_cardiomegaly \
  --split-col split \
  --train-split train \
  --val-split validate \
  --test-split test \
  --subgroup-cols age_bin \
  --search-method exhaustive \
  --lambda-y 0.0 \
  --lambda-r 0.0


for lam in 0.0 0.0001 0.0003 0.001 0.003 0.005 0.01 0.02; do
  python -m causal_analysis \
    --data-path results/pleural_effusion/mimic_cxr/full_finetune/resnet50/seed_42/predictions.csv \
    --output-dir /output_debug_exact/mimic/pleural_effusion/resnet50_FT/Metadata \
    --dataset-key mimic_cxr_chest \
    --gt-col gt_pleural_effusion \
    --prob-col prob_pleural_effusion \
    --split-col split \
    --train-split train \
    --val-split validate \
    --test-split test \
    --subgroup-cols age_bin \
    --search-method exhaustive \
    --lambda-y "${lam}" \
    --lambda-r "${lam}"
done

for lam in 0.0 0.0001 0.0003 0.001 0.003 0.005 0.01 0.02; do
  python -m causal_analysis \
    --data-path results/pneumothorax/mimic_cxr/full_finetune/resnet50/seed_42/predictions.csv \
    --output-dir /output_debug_exact/mimic/pneumothorax/resnet50_FT/Metadata \
    --dataset-key mimic_cxr_chest \
    --gt-col gt_pneumothorax \
    --prob-col prob_pneumothorax \
    --split-col split \
    --train-split train \
    --val-split validate \
    --test-split test \
    --subgroup-cols age_bin \
    --search-method exhaustive \
    --lambda-y "${lam}" \
    --lambda-r "${lam}"
done



for lam in 0.0 0.0001 0.0003 0.001 0.003 0.005 0.01 0.02; do
  python -m causal_analysis \
    --data-path results/cardiomegaly/mimic_cxr/full_finetune/resnet50/seed_42/predictions.csv \
    --output-dir /output_debug_exact/mimic/cardiomegaly/resnet50_FT/Metadata \
    --dataset-key mimic_cxr_chest \
    --gt-col gt_cardiomegaly \
    --prob-col prob_cardiomegaly \
    --split-col split \
    --train-split train \
    --val-split validate \
    --test-split test \
    --subgroup-cols age_bin \
    --search-method exhaustive \
    --lambda-y "${lam}" \
    --lambda-r "${lam}"
done




python -m causal_analysis \
  --data-path /Ladder/out_HF/out/Ladder/out/RSNA/fold0/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/test_abnormal_dataframe_mitigation_with_dicom_headers_v1.csv \
  --output-dir /output_debug_exact/RSNA/cancer/enb5/All \
  --dataset-key rsna_mammo \
  --gt-col out_put_GT \
  --prob-col out_put_predict \
  --subgroup-cols site_id_bin \
  --search-method-stage1 exhaustive \
  --search-method-stage2 all \
   --lambda-y "${lam}" \
    --lambda-r "${lam}"


python -m causal_analysis \
    --data-path results/cardiomegaly/mimic_cxr/full_finetune/resnet50/seed_42/predictions.csv \
    --output-dir /output/mimic/cardiomegaly/resnet50_FT/Metadata \
    --dataset-key mimic_cxr_chest \
    --gt-col gt_cardiomegaly \
    --prob-col prob_cardiomegaly \
    --split-col split \
    --train-split train \
    --val-split validate \
    --test-split test \
    --subgroup-cols race_bin \
    --search-method exhaustive \
    --lambda-y 0 \
    --lambda-r 0

python -m causal_analysis \
    --data-path results/cardiomegaly/mimic_cxr/full_finetune/resnet50/seed_42/predictions.csv \
    --output-dir /output/mimic/cardiomegaly/resnet50_FT/Metadata \
    --dataset-key mimic_cxr_chest \
    --gt-col gt_cardiomegaly \
    --prob-col prob_cardiomegaly \
    --split-col split \
    --train-split train \
    --val-split validate \
    --test-split test \
    --subgroup-cols age_bin \
    --search-method-stage1 all  \
    --search-method-stage2 exhaustive \
    --lambda-y 0 \
    --lambda-r 0


python -m causal_analysis \
    --data-path /Ladder/out_HF/out/Ladder/out/ViNDr/fold0/clip_img_encoder_tf_efficientnet_b5_ns-detect/test_abnormal_dataframe_mitigation_with_dicom_headers.csv \
    --output-dir /output_debug_exact/VinDr/cancer/enb5/All \
    --dataset-key vindr_mammo \
    --gt-col out_put_GT \
    --prob-col out_put_predict \
    --split-col split \
    --train-split train \
    --val-split validate \
    --test-split test \
    --subgroup-cols "model_name_bin" \
    --search-method-stage1 exhaustive \
    --search-method-stage2 all \
    --lambda-y 0 \
    --lambda-r 0

python -m causal_analysis \
    --data-path /mb3/results/mimic_cxr/full_finetune/densenet121/seed_42/mimic_cardio_densenet121/predictions.csv \
    --output-dir /output/mimic/cardiomegaly/densenet121_FT/Metadata \
    --dataset-key mimic_cxr_chest \
    --gt-col gt_cardiomegaly \
    --prob-col prob_cardiomegaly \
    --split-col split \
    --train-split train \
    --val-split validate \
    --test-split test \
    --subgroup-cols age_bin \
    --search-method-stage1 all  \
    --search-method-stage2 exhaustive \
    --lambda-y 0 \
    --lambda-r 0


python -m causal_analysis.run_r_controls \
    --data-path /Ladder/out_HF/out/Ladder/out/RSNA/fold0/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/test_abnormal_dataframe_mitigation_with_dicom_headers_v1.csv \
    --output-dir /output/RSNA/cancer/enb5/All/lam_0.0 \
    --dataset-key rsna_mammo \
    --gt-col out_put_GT \
    --prob-col out_put_predict \
    --subgroup-cols site_id_bin \
    --search-method-dir exhaustive