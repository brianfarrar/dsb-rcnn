from __future__ import division

# import libraries
import logging
import argparse
import dsbhelper
import os
import cv2
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from skimage.segmentation import random_walker, active_contour
from skimage.filters import gaussian, median
from skimage import measure


# ---------------------------------------------------------------------
# save new masks
# ---------------------------------------------------------------------
def save_new_mask(args, specimen_id, mask, mask_sub_folder, mask_fname):

    specimen_folder_name = specimen_id
    folder = args.train_folder + '/' + specimen_folder_name + '/' + mask_sub_folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    cv2.imwrite(folder + '/' + mask_fname, mask)


# ---------------------------------------------------------------------
# Creates a new mask using watershed methodology
# inspired by: Allen Goodman in this post:
# https://www.kaggle.com/c/data-science-bowl-2018/discussion/48130
# ---------------------------------------------------------------------
def watershed_mask(mask):

    # dilate the mask to get sure background
    kernel = np.ones((2, 2), np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=1)

    # threshold with a distance transform to get the sure foreground
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Find the unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label with markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Add one to all labels so that sure background is not 0, but 1
    markers[unknown == 255] = 0  # Now, mark the region of unknown with zero

    # compute the watershed image
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    new_mask = cv2.watershed(mask, markers)
    new_mask[new_mask == -1] = 1
    new_mask = new_mask - 1

    if new_mask.max() == 1:
        new_mask = new_mask * 255

    return new_mask


# ---------------------------------------------------------------------
# Creates a new mask using random walk methodology
# inspired by: Allen Goodman in this post:
# https://www.kaggle.com/c/data-science-bowl-2018/discussion/48130
# ---------------------------------------------------------------------
def random_walk_mask(mask):
    # dilate the mask to get sure background
    kernel = np.ones((2, 2), np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=1)

    # do a distance transformation on the mask
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)

    # get the centroid pixel value
    centroid = center_of_mass(dist_transform)
    centroid_pixel_value = dist_transform[int(centroid[0])][int(centroid[1])]

    # get markers for the random_walk mask
    markers = np.zeros(mask.shape, dtype=np.uint8)
    markers[dist_transform >= 0.2 * centroid_pixel_value] = 2
    markers[dist_transform < 0.1 * centroid_pixel_value] = 1

    new_mask = random_walker(sure_bg, markers)
    new_mask = new_mask - 1
    if new_mask.max() == 1:
        new_mask = (new_mask * 255).astype(np.uint8)
    else:
        new_mask = new_mask.astype(np.uint8)

    return new_mask


# ---------------------------------------------------------------------
# Creates a new mask using active contours methodology
# inspired by: Allen Goodman in this post:
# https://www.kaggle.com/c/data-science-bowl-2018/discussion/48130
# ---------------------------------------------------------------------
def active_contour_mask(mask, blur_fn='gaussian'):
    # get the centroid of this mask
    centroid = center_of_mass(mask)

    # get a bounding circle
    s = np.linspace(0, 2 * np.pi, 400)
    x = int(centroid[1]) + 30 * np.cos(s)
    y = int(centroid[0]) + 30 * np.sin(s)
    base_circle = np.array([x, y]).T

    if blur_fn == 'gaussian':
        blurred_mask = gaussian(mask, 2)
    elif blur_fn == 'median':
        kernel = np.ones((2, 2), np.uint8)
        blurred_mask = median(mask, kernel)
    else:
        print('{} blur function not implemented.  Try gaussian...'.format(blur_fn))

    # calculate the active contour
    contour = active_contour(blurred_mask, base_circle, alpha=0.009, beta=1000, gamma=0.001)

    # note: contour returns the horizontal coordinate first, the cartesian 'x'
    # this is different than numpy's shape, and center_of_mass which provides
    # the vertical (cartesian 'y') first
    rectified_contour = []
    for coord in contour:
        if coord[0] < 0:
            coord[0] = 0
        elif coord[0] > mask.shape[1]:
            coord[0] = mask.shape[1]
        if coord[1] < 0:
            coord[1] = 0
        elif coord[1] > mask.shape[0]:
            coord[1] = mask.shape[0]
        rectified_contour.append(coord)

    contour = np.asarray(rectified_contour)

    # convert contour to a mask (see note above for explanation re: np.fliplr below)
    contour = np.fliplr(contour)
    new_mask = measure.grid_points_in_poly(mask.shape, contour).astype(np.uint8)

    new_mask = new_mask * 255

    return new_mask


# ---------------------------------------------------------------------
# make masks with automated annotations
# ---------------------------------------------------------------------
def make_masks(args):

    # get a list of all the specimens
    specimen_ids = dsbhelper.get_specimen_ids(args.train_folder)

    # use this to test local on small batches
    #specimen_ids = specimen_ids[:32]

    for specimen_id in specimen_ids:

        print('Processing specimen id: {}'.format(specimen_id[0:5]))

        # get all of the masks for this specimen
        masks = dsbhelper.get_specimen_masks(args.train_folder, specimen_id)
        mask_fname_list = dsbhelper.get_specimen_mask_list(args.train_folder, specimen_id)

        for i, mask in enumerate(masks):

            print('Processing mask #{}'.format(i))

            # create watershed mask
            ws_mask = watershed_mask(mask)
            save_new_mask(args, specimen_id, ws_mask, 'mask_ws', mask_fname_list[i])
            print(' -> watershed mask saved')

            # create random walk mask
            rw_mask = random_walk_mask(mask)
            rw_mask = watershed_mask(rw_mask)
            save_new_mask(args, specimen_id, rw_mask, 'mask_rw', mask_fname_list[i])
            print(' -> random walk mask saved')

            # create active contours mask using gaussian blur
            ac_gblur_mask = active_contour_mask(mask, blur_fn='gaussian')
            ac_gblur_mask = watershed_mask(ac_gblur_mask)
            save_new_mask(args, specimen_id, ac_gblur_mask, 'mask_ag', mask_fname_list[i])
            print(' -> ac gaussian mask saved')

            # create active contours mask using median blur
            ac_mblur_mask = active_contour_mask(mask, blur_fn='median')
            ac_mblur_mask = watershed_mask(ac_mblur_mask)
            save_new_mask(args, specimen_id, ac_mblur_mask, 'mask_am', mask_fname_list[i])
            print(' -> ac median mask saved')


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

    # make masks
    make_masks(args)


def main():
    logging.getLogger().setLevel(logging.INFO)
    run()


if __name__ == '__main__':
    main()
