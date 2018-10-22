#!/usr/bin/env bash
gsutil -m cp trainer/*.py gs://mwpdsb/mask_rcnn/code
gsutil -m cp *.sh gs://mwpdsb/mask_rcnn/code
gsutil -m cp trainer/*.ipynb gs://mwpdsb/mask_rcnn/code