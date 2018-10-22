#!/usr/bin/env bash
python3 copy_tensorboard_to_gcs.py \
          --model_folder ./model \
          --gcs_log_folder gs://mwpdsb/mask_rcnn/tflogs/tflogs_a43f5c
