#!/usr/bin/env bash
gsutil -m cp gs://mwpdsb/mask_rcnn/code/*.py ./
gsutil -m cp gs://mwpdsb/mask_rcnn/code/*.sh ./
gsutil -m cp gs://mwpdsb/mask_rcnn/code/*.ipynb ./
chmod u+x copy_dsb_mrcnn_from_gcs.sh
chmod u+x run_dsb_mask_rcnn_gce.sh
chmod u+x run_augment_train_data.sh
chmod u+x run_grayscale_train_data.sh
chmod u+x run_tensorboard_to_gcs.sh
chmod u+x run_make_new_masks.sh