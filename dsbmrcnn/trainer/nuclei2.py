import random
#import cv2
import math
import numpy as np
import os
import logging
from random import randint

__runtype__ = 'cloud'

if __runtype__ is 'local':
    from . import config
    from . import utils
    from . import dsbhelper
else:
    os.sys.path.append(os.path.dirname(os.path.abspath('.')))
    import config
    import utils
    import dsbhelper


# --------------------------------------------------------------------------------------------------
# NucleiConfig is a modification of the Config base class to support Data Science Bowl training
# --------------------------------------------------------------------------------------------------
class NucleiConfig(config.Config):

    # Give the configuration a recognizable name
    NAME = "nuclei"

    # set GPUs and number of images per
    # this effectively sets the batch size
    GPU_COUNT = 8
    IMAGES_PER_GPU = 2

    # There are 664 base images, split them 604 to train 60 to validation
    STEPS_PER_EPOCH = int(math.ceil(3260 / (IMAGES_PER_GPU * GPU_COUNT)))

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = int(math.ceil(60 / (IMAGES_PER_GPU * GPU_COUNT)))


    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 nuclei

    # anchor size in pixels
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 400

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Determines image shape
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # do not zero center
    MEAN_PIXEL = [0, 0, 0]

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 500

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 500

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # training has 3 stages, learning rate for each here
    LEARNING_RATE = 1e-3
    LEARNING_RATE_2 = 1e-4
    LEARNING_RATE_3 = 1e-5
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001


    # training has 3 stages, epoch count for each here
    EPOCHS = 10
    EPOCHS_2 = 5
    EPOCHS_3 = 5


# --------------------------------------------------------------------------------------------------
# NucleiDataset is a modification of the Dataset base class to support Data Science Bowl data
# --------------------------------------------------------------------------------------------------

class NucleiDataset(utils.Dataset):

    # size of the random crop
    CROP_SIZE = 128

    # returns a cropped image
    def crop_image(self, image, x, y):
        return image[x:x + self.CROP_SIZE, y:y + self.CROP_SIZE, :]

    # returns a cropped mask
    def crop_mask(self, image, x, y):
        return image[x:x + self.CROP_SIZE, y:y + self.CROP_SIZE]


    # ------------------------------------------------------------------------
    # this method loads metadata about train or test images
    # ------------------------------------------------------------------------
    def load_data(self, path, mode='train', filter_ids=None):

        self.add_class("nucleis", 1, "nuclei")

        specimen_ids = dsbhelper.get_specimen_ids(path)

        # this handles train, validate, mini-train, and mini-validate mode
        # if mode is 'test' then use all the images in path
        if filter_ids is not None:
            if mode is 'train':
                specimen_ids = [specimen_id for specimen_id in specimen_ids if specimen_id not in filter_ids]
            elif mode in ['validate', 'mini-train', 'mini-validate']:
                specimen_ids = [item for item in specimen_ids if item in filter_ids]
            else:
                logging.warning('Invalid mode {} specified'.format(mode))

        for i, specimen_id in enumerate(specimen_ids):

            bg_color = random.randint(0, 255)

            image = dsbhelper.get_specimen_image(path, specimen_id)

            if mode is 'train':

                height = self.CROP_SIZE
                width = self.CROP_SIZE
                full_height, full_width, channels = image.shape

                # get a random x for the origin that is greater than CROP_SIZE from the right edge
                crop_origin_x = randint(0, full_width - self.CROP_SIZE)
                crop_origin_y = randint(0, full_height - self.CROP_SIZE)

            elif mode is 'validate':

                height, width, channels = image.shape
                crop_origin_x = 0
                crop_origin_y = 0

            self.add_image("nucleis",
                           image_id=i,
                           path=path,
                           width=width,
                           height=height,
                           crop_origin_x=crop_origin_x,
                           crop_origin_y=crop_origin_y,
                           bg_color=bg_color,
                           specimen_id=specimen_id,
                           mode=mode)

    # -----------------------------
    # this method loads an image
    # -----------------------------
    def load_image(self, image_id):

        info = self.get_info(image_id)
        path = info['path']
        specimen_id = info['specimen_id']
        x = info['crop_origin_x']
        y = info['crop_origin_y']
        mode = info['mode']
        i = dsbhelper.get_specimen_image(path, specimen_id)

        # return cropped image or if validating or generating submission return whole image
        if mode is 'train':
            image = self.crop_image(i, x, y)
        elif mode is 'validate':
            image = i

        return image

    # -----------------------------
    # this method loads a mask set
    # -----------------------------
    def load_mask(self, image_id):

        info = self.get_info(image_id)
        path = info['path']
        specimen_id = info['specimen_id']
        mode = info['mode']
        height = info['height']
        width = info['width']
        x = info['crop_origin_x']
        y = info['crop_origin_y']
        mode = info['mode']

        if mode not in ['train', 'validate', 'mini-train', 'mini-validate']:
            logging.warning('Invalid mode {} specified!'.format(mode))
        else:
            mask_images = dsbhelper.get_specimen_masks(path, specimen_id)

        if mode is 'train':
            mask_images = [self.crop_mask(mask, x, y) for mask in mask_images]

        # first eliminate bad masks from the count
        label_count = 0
        for i, mask_image in enumerate(mask_images):
            if np.sum(mask_image) == 0:
                logging.info('Specimen ID: {}, mask #: {} is all zeros!'.format(specimen_id, i))
            else:
                label_count += 1

        # now make the mask array with the correct number of entries
        masks = np.zeros((height, width, label_count), dtype=np.bool)

        # reset the label counter and loop again to make the array of masks and classes
        label_count = 0
        for i, mask_image in enumerate(mask_images):
            if np.sum(mask_image) != 0:
                masks[:, :, label_count] = mask_image
                label_count += 1

        # now make the class id array
        class_ids = np.array([1] * label_count)
        class_ids = class_ids.astype(np.int32)

        return masks, class_ids

    # -------------------------------------
    # returns image info for an image id
    #--------------------------------------
    def get_info(self, image_id):
        return self.image_info[image_id]


# configure for inference
class InferenceConfig(NucleiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
