from __future__ import division

# import libraries
import numpy as np
import logging
import argparse
import dsbhelper
import cv2
import os


# ---------------------------------------------------------------------
# save augmented image
# ---------------------------------------------------------------------
def save_aug_image(args, specimen_id, aug_label, image):

    folder_name = specimen_id + '_' + aug_label
    fname = folder_name + '.png'

    if not os.path.exists(args.train_folder + '/' + folder_name):
        folder = args.train_folder + '/' + folder_name
        os.makedirs(folder)
    if not os.path.exists(args.train_folder + '/' + folder_name + '/images'):
        folder = args.train_folder + '/' + folder_name + '/images'
        os.makedirs(folder)

    cv2.imwrite(args.train_folder + '/' + folder_name + '/images/' + fname, image)


# ---------------------------------------------------------------------
# save augmented masks
# ---------------------------------------------------------------------
def save_aug_masks(args, specimen_id, aug_label, masks):

    folder_name = specimen_id + '_' + aug_label
    mask_filename_list = dsbhelper.get_specimen_mask_list(args.train_folder, specimen_id)

    if not os.path.exists(args.train_folder + '/' + folder_name):
        folder = args.train_folder + '/' + folder_name
        os.makedirs(folder)
    if not os.path.exists(args.train_folder + '/' + folder_name + '/masks'):
        folder = args.train_folder + '/' + folder_name + '/masks'
        os.makedirs(folder)

    for i in range(len(masks)):
        fname = mask_filename_list[i].split('.')[0] + '_' + aug_label + '.png'
        cv2.imwrite(args.train_folder + '/' + folder_name + '/masks/' + fname, masks[i])


# ---------------------------------------------------------------------
# get training data
# ---------------------------------------------------------------------
def get_train_data(args):

    # get a list of all the specimens
    specimen_ids = dsbhelper.get_specimen_ids(args.train_folder)

    # use this to test local on small batches
    #specimen_ids = specimen_ids[:32]

    train_set_size = len(specimen_ids)

    for j, specimen_id in enumerate(specimen_ids):

        logging.info('Processing augmentation {} of {}, train specimen id: {}'.format(j + 1, train_set_size, specimen_id[:5]))


        # get a specimen image
        image = dsbhelper.get_specimen_image(args.train_folder, specimen_id)

        i = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image[:, :, 0] = i
        image[:, :, 1] = i
        image[:, :, 2] = i

        cv2.imwrite(args.train_folder + '/' + specimen_id + '/images/' + specimen_id + '.png', image)


# ----------------------------------------------------------------------------------------------------------------
#
# Function: run(argv):          Runs the data generator.  This is the main runner for this package.
#
# Arguments:                    See below, but all of them have a default
#
# Returns:                      None
#
# ----------------------------------------------------------------------------------------------------------------

def run(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_folder', dest='train_folder', default='./data/stage1_train',
                        help='Folder with training images and masks')

    # get the command line arguments
    args, _ = parser.parse_known_args(argv)

    # make train data
    get_train_data(args)


def main():
    logging.getLogger().setLevel(logging.INFO)
    run()

if __name__ == '__main__':
    main()
