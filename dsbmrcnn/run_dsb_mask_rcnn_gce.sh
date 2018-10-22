#!/usr/bin/env bash
declare -r dsb_job_name='mwp_dsb_mask_rcnn_'$(date +"%Y%m%d_%H%M%S")
declare -r model='mask_rcnn'
echo "--------------------------------------"
echo "job name:  "$dsb_job_name
echo "model:     "$model
echo "--------------------------------------"

python3 task.py \
          --run_training "False" \
          --run_eval "False" \
          --run_ensemble_eval "False" \
          --run_predict "False" \
          --run_ensemble_predict "True" \
          --new_model "False" \
          --model_name $model \
          --model_folder gs://mwpdsb/mask_rcnn/models/model_b12eb9 \
          --pretrained_weights_name "last" \
          --pretrained_weights_folder gs://mwpdsb/mask_rcnn/pretrained \
          --output_folder gs://mwpdsb/mask_rcnn/outputs/output \
          --submission_folder gs://mwpdsb/mask_rcnn/submission \
          --zip_folder gs://mwpdsb/data \
          --job_id $dsb_job_name
