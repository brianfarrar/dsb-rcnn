import random
import cv2
import math
import numpy as np
import os
import logging
from skimage.transform import resize

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

    # There are 664 base images (fixed data set), split them 604 to train 60 to validation
    # There are 670 base images (orig data set), split them 610 to train 60 to validation
    STEPS_PER_EPOCH = int(math.ceil(773 / (IMAGES_PER_GPU * GPU_COUNT)))
    #STEPS_PER_EPOCH =  int(math.ceil(531 / (IMAGES_PER_GPU * GPU_COUNT)))
    #STEPS_PER_EPOCH = int(math.ceil((5*531) / (IMAGES_PER_GPU * GPU_COUNT)))

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = int(math.ceil(133 / (IMAGES_PER_GPU * GPU_COUNT)))

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64, 128]

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 nuclei

    # anchor size in pixels
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64) # got LB High with these settings

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
    RPN_TRAIN_ANCHORS_PER_IMAGE = 320 # Got LB high with this setting
    #RPN_TRAIN_ANCHORS_PER_IMAGE = 1000

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 2000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True  # Got LB high with this setting
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask # Got LB high with this setting


    # Determines image shape
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # zero center
    MEAN_PIXEL = np.array([42.17746161, 38.21568456, 46.82167803])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 512

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 512

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 1000

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    # DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_MIN_CONFIDENCE = 0.5

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # training has 3 stages, learning rate for each here
    LEARNING_RATE = 1e-4
    LEARNING_RATE_2 = 1e-4
    LEARNING_RATE_3 = 1e-5
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # training has 3 stages, epoch count for each here
    EPOCHS = 60
    EPOCHS_2 = 40
    EPOCHS_3 = 0


# --------------------------------------------------------------------------------------------------
# NucleiDataset is a modification of the Dataset base class to support Data Science Bowl data
# --------------------------------------------------------------------------------------------------

class NucleiDataset(utils.Dataset):

    # size of the random crop
    CROP_SIZE = 512

    # true if edges should be mirrored
    USE_MIRRORS = False

    # ---------------------------------------------
    # returns a cropped image
    # ----------------------------------------------
    def crop_image(self, image, h, w):

        original_h, original_w, _ = image.shape

        # crop the sides only if necessary
        h_length = min([original_h, self.CROP_SIZE])
        w_length = min([original_w, self.CROP_SIZE])

        return image[h:h + h_length, w:w + w_length, :]

    # ----------------------------------------------
    # returns a cropped mask
    # ----------------------------------------------
    def crop_mask(self, image, h, w):

        original_h, original_w = image.shape

        # crop the sides only if necessary
        h_length = min([original_h, self.CROP_SIZE])
        w_length = min([original_w, self.CROP_SIZE])

        return image[h:h + h_length, w:w + w_length]

    # ------------------------------------------------------------------------
    # this method loads metadata about train or test images
    # ------------------------------------------------------------------------
    def load_data(self, path, mode='train', filter_ids=None):

        # add a class for nuclei (see matterport docs)
        self.add_class("nucleis", 1, "nuclei")

        # get the list of specimen_ids
        specimen_ids = dsbhelper.get_specimen_ids(path, mode='train_set')

        # this handles train, validate, mini-train, and mini-validate mode
        # if mode is 'test' then use all the images in path
        if filter_ids is not None:
            if mode is 'train':
                specimen_ids = [specimen_id for specimen_id in specimen_ids if specimen_id not in filter_ids]
            elif mode in ['validate', 'mini-train', 'mini-validate', 'cluster-focus']:
                specimen_ids = [item for item in specimen_ids if item in filter_ids]
            elif mode is 'test':
                pass
            else:
                logging.warning('Invalid mode {} specified'.format(mode))

        # loop through each specimen_id
        for i, specimen_id in enumerate(specimen_ids):

            # set a random background color
            bg_color = random.randint(0, 255)

            # get an image and initialize crop dimensions.
            # crop dimensions only used if the image is bigger than self.CROP_SIZE
            # see load_image below
            image = dsbhelper.get_specimen_image(path, specimen_id)
            crop_origin_h = 0
            crop_origin_w = 0

            # mirror the edges if desired
            # this only matters here so that dimensions will calculate properly
            if self.USE_MIRRORS:
                image = dsbhelper.mirror_edges(image)

            # get full size of image
            full_height = image.shape[0]
            full_width = image.shape[1]

            # add the image and meta data to the dataset
            self.add_image("nucleis",
                           image_id=i,
                           path=path,
                           full_height=full_height,
                           full_width=full_width,
                           crop_origin_h=crop_origin_h,
                           crop_origin_w=crop_origin_w,
                           bg_color=bg_color,
                           specimen_id=specimen_id,
                           aug='none',
                           mode=mode)

    # -----------------------------
    # this method loads an image
    # -----------------------------
    def load_image(self, image_id):

        # get meta data
        info = self.get_info(image_id)
        path = info['path']
        specimen_id = info['specimen_id']
        mode = info['mode']
        full_height = info['full_height']
        full_width = info['full_width']
        crop_origin_h = info['crop_origin_h']
        crop_origin_w = info['crop_origin_w']

        # flag to signal crop op
        crop_image = False

        # get an image
        image = dsbhelper.get_specimen_image(path, specimen_id)

        # mirror the edges if desired
        if self.USE_MIRRORS:
            image = dsbhelper.mirror_edges(image)

        # turn cropping on if necessary for this image
        if full_height > self.CROP_SIZE or full_width > self.CROP_SIZE:
            crop_image = True

        # crop if necessary
        if crop_image == True and mode == 'train':

            pixel_sum = 0
            while pixel_sum == 0:

                # get a random x for the origin that is greater than CROP_SIZE from the right edge/bottom edge
                if full_width > self.CROP_SIZE:
                    crop_origin_w = random.randint(0, full_width - self.CROP_SIZE)
                else:
                    crop_origin_w = 0

                if full_height > self.CROP_SIZE:
                    crop_origin_h = random.randint(0, full_height - self.CROP_SIZE)
                else:
                    crop_origin_h = 0

                crop = self.crop_image(image, crop_origin_h, crop_origin_w)
                pixel_sum = crop.sum()

            # now that we have a valid crop, set it equal to image
            image = crop

            # update meta data
            self.image_info[image_id]['crop_origin_w'] = crop_origin_w
            self.image_info[image_id]['crop_origin_h'] = crop_origin_h

        # set up augmentations
        #aug_list = ['gray', 'flip', 'color', 'noise', 'blur']
        aug_list = ['gray', 'color', 'noise', 'blur']
        aug = 'none'

        # only do augmentation in train mode
        if mode == 'train':

            # only augment 10% of the images
            if np.random.uniform() > 0.9:
                aug = random.choice(aug_list)
                self.image_info[image_id]['aug'] = aug

        # augment the images
        if aug == 'none':
            pass

        elif aug == 'gray':
            i = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image[:, :, 0] = i
            image[:, :, 1] = i
            image[:, :, 2] = i

        elif aug == 'flip':
            image = np.fliplr(image)

        elif aug == 'color':
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h = hsv[:, :, 0]
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            shift = random.randint(5, 200)
            recolor = np.zeros_like(image)
            recolor[:, :, 0] = h + shift
            recolor[:, :, 1] = s + shift
            recolor[:, :, 2] = v
            image = cv2.cvtColor(recolor, cv2.COLOR_HSV2RGB)

        elif aug == 'noise':
            height, width, channels = image.shape
            noise = np.random.randint(0, 50, (height, width))
            jitter = np.zeros_like(image)
            jitter[:, :, 1] = noise
            image = cv2.add(image, jitter)

        elif aug == 'blur':
            image = cv2.blur(image, (5, 5))

        return image

    # -----------------------------
    # this method loads a mask set
    # -----------------------------
    def load_mask(self, image_id):

        # get metadata
        info = self.get_info(image_id)
        path = info['path']
        specimen_id = info['specimen_id']
        mode = info['mode']
        full_height = info['full_height']
        full_width = info['full_width']
        crop_origin_w = info['crop_origin_w']
        crop_origin_h = info['crop_origin_h']
        aug = info['aug']
        crop_mask = False

        # check mode and load mask_images
        if mode not in ['train', 'validate', 'mini-train', 'mini-validate', 'cluster-focus']:
            logging.warning('Invalid mode {} specified!'.format(mode))
        else:
            mask_images = dsbhelper.get_specimen_masks(path, specimen_id)
            actual_height = full_height
            actual_width = full_width

        # first eliminate bad masks from the count
        label_count = 0
        for i, mask_image in enumerate(mask_images):

            # mirror edges if desired
            if self.USE_MIRRORS:
                mask_image = dsbhelper.mirror_edges(mask_image, is_mask=True)

            # turn cropping on if necessary for this image
            if full_height > self.CROP_SIZE or full_width > self.CROP_SIZE:
                crop_mask = True

            # crop if desired
            if crop_mask == True and mode == 'train':
                mask_image = self.crop_mask(mask_image, crop_origin_h, crop_origin_w)
                actual_height = min([full_height, self.CROP_SIZE])
                actual_width = min([full_width, self.CROP_SIZE])

            if np.sum(mask_image) == 0:
                #logging.info('Specimen ID: {}, mask #: {} is all zeros!'.format(specimen_id, i))
                pass
            else:
                label_count += 1

        # now make the mask array with the correct number of entries
        masks = np.zeros((actual_height, actual_width, label_count), dtype=np.bool)

        # reset the label counter and loop again to make the array of masks and classes
        label_count = 0
        for i, mask_image in enumerate(mask_images):

            # of the augmentation was to flip, then we have to flip the masks before returning them
            #if aug == 'flip':
            #    mask_image = np.fliplr(mask_image)

            # mirror if desired
            if self.USE_MIRRORS:
                mask_image = dsbhelper.mirror_edges(mask_image, is_mask=True)

            # turn cropping on if necessary for this image
            if full_height > self.CROP_SIZE or full_width > self.CROP_SIZE:
                crop_mask = True

            # crop if desired
            if crop_mask == True and mode == 'train':
                mask_image = self.crop_mask(mask_image, crop_origin_h, crop_origin_w)

            if np.sum(mask_image) != 0:
                masks[:, :, label_count] = mask_image
                label_count += 1

        # now make the class id array
        class_ids = np.array([1] * label_count)
        class_ids = class_ids.astype(np.int32)

        return masks, class_ids

    # -------------------------------------
    # returns image info for an image id
    # --------------------------------------
    def get_info(self, image_id):
        return self.image_info[image_id]


# configure for inference
class InferenceConfig(NucleiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1