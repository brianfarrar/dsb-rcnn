#!/usr/bin/env bash
declare -r dsb_job_name='mwp_dsb_mask_rcnn_'$(date +"%Y%m%d_%H%M%S")
declare -r model='dsb_mask_rcnn'
echo "--------------------------------------"
echo "job name:  "$dsb_job_name
echo "model:     "$model
echo "--------------------------------------"

#source /Users/farrar/py2/bin/activate.sh

declare -r TRAIN_CLOUD=true

if (${TRAIN_CLOUD}); then
  echo "train on cloud"
  gcloud ml-engine jobs submit training $dsb_job_name \
          --config config.yaml \
          --package-path /Users/farrar/PycharmProjects/dsb-rcnn/dsbmrcnn/trainer \
          --module-name trainer.task \
          --job-dir gs://mwpdsb/train \
          --staging-bucket gs://mwpdsb-staging \
          --region us-central1 \
          --runtime-version 1.4 \
          -- \
          --model_name $model \
          --model_folder gs://mwpdsb/mask_rcnn/model \
          --pretrained_weights gs://mwpdsb/shapes/pretrained/mask_rcnn_coco.h5 \
          --output_folder gs://mwpdsb/mask_rcnn/output \
          --zip_folder gs://mwpdsb/data/zips \
          --epochs 10 \
          --batch_size 128 \
          --eval_split 0.1 \
          --gpu_count 8 \
          --job_id $dsb_job_name
else
  echo "Testing variables"
fi