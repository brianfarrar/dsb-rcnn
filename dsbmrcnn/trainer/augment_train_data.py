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

    train_set_size = len(specimen_ids) * 4

    # set the example_counter to 0
    example_counter = 0

    for specimen_id in specimen_ids:

        # get a specimen image
        image = dsbhelper.get_specimen_image(args.train_folder, specimen_id)

        # get all of the masks for this specimen
        masks = dsbhelper.get_specimen_masks(args.train_folder, specimen_id)

        # rotate 90
        logging.info('Processing augmentation {} of {}, train specimen id: {}'.format(example_counter + 1, train_set_size,
                                                                                      specimen_id[:5]))
        aug = np.rot90(image, 1)
        save_aug_image(args, specimen_id, 'rot090', aug)

        aug_masks = [np.rot90(mask, 1) for mask in masks]
        save_aug_masks(args, specimen_id, 'rot090', aug_masks)

        example_counter += 1

        # rotate 180
        logging.info('Processing augmentation {} of {}, train specimen id: {}'.format(example_counter + 1, train_set_size,
                                                                                      specimen_id[:5]))
        aug = np.rot90(image, 2)
        save_aug_image(args, specimen_id, 'rot180', aug)

        aug_masks = [np.rot90(mask, 2) for mask in masks]
        save_aug_masks(args, specimen_id, 'rot180', aug_masks)

        example_counter += 1

        # rotate 270
        logging.info('Processing augmentation {} of {}, train specimen id: {}'.format(example_counter + 1, train_set_size,
                                                                                      specimen_id[:5]))
        aug = np.rot90(image, 3)
        save_aug_image(args, specimen_id, 'rot270', aug)

        aug_masks = [np.rot90(mask, 3) for mask in masks]
        save_aug_masks(args, specimen_id, 'rot270', aug_masks)

        example_counter += 1

        # flip horizontally
        logging.info('Processing augmentation {} of {}, train specimen id: {}'.format(example_counter + 1, train_set_size,
                                                                                      specimen_id[:5]))
        aug = np.fliplr(image)
        save_aug_image(args, specimen_id, 'fliplr', aug)

        aug_masks = [np.fliplr(mask) for mask in masks]
        save_aug_masks(args, specimen_id, 'fliplr', aug_masks)
        example_counter += 1

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
