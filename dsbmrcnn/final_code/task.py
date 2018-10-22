import logging
import argparse
import json
import os
import uuid
import numpy as np

import matplotlib
matplotlib.use('Agg')

__runtype__ = 'local'

if __runtype__ is 'cloud':
    from . import nuclei
    from . import model as modellib
    from . import dsbhelper
    from . import utils
else:
    os.sys.path.append(os.path.dirname(os.path.abspath('.')))
    import nuclei
    import model as modellib
    import dsbhelper
    import utils


# ----------------------------------------------------------------------------------------------------------------
#
# Function: run(argv):          Runs the model.  This is the main runner for this package.
#
# Arguments:                    See below, but all of them have a default
#
# Returns:                      None
#
# ----------------------------------------------------------------------------------------------------------------

def run(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_training', dest='run_training', default='True',
                        help='Text boolean to decide whether to run training')

    parser.add_argument('--run_eval', dest='run_eval', default='True',
                        help='Text boolean to decide whether to run eval')

    parser.add_argument('--run_predict', dest='run_predict', default='True',
                        help='Text boolean to decide whether to run predict')

    parser.add_argument('--new_model', dest='new_model', default='True',
                        help='Text boolean to decide whether to run predict')

    parser.add_argument('--model_name', dest='model_name', default='mask_rcnn',
                        help='Model to run.')

    parser.add_argument('--model_folder', dest='model_folder', default='./',
                        help='Folder to store the model checkpoints')

    parser.add_argument('--pretrained_weights_folder', dest='pretrained_weights_folder', default='./pretrained',
                        help='Folder to retrieve pretrained weights from')

    parser.add_argument('--pretrained_weights_name', dest='pretrained_weights_name', default='imagenet',
                        help='imagenet, coco, or last')

    parser.add_argument('--output_folder', dest='output_folder', default='./output',
                        help='Folder to output images to')

    parser.add_argument('--submission_folder', dest='submission_folder', default='./submission',
                        help='Folder for submission csv')

    parser.add_argument('--zip_folder', dest='zip_folder', default='./zips',
                        help='Folder that contains the zipped data files')

    parser.add_argument('--job_id', dest='job_id', default='',
                        help='Unique job id')

    # get the command line arguments
    args, _ = parser.parse_known_args(argv)

    # If we are starting a new model, then get a new unique id otherwise, get it from the model folder name
    if dsbhelper.text_to_bool(args.new_model):
        unique_id = str(uuid.uuid4())[-6:]
        logging.info('New model number -> {}'.format(unique_id))
        args.model_folder = '{}_{}'.format(args.model_folder, unique_id)
        args.output_folder = '{}_{}'.format(args.output_folder, unique_id)
    else:
        unique_id = args.model_folder.split('_')[-1]
        args.output_folder = '{}_{}'.format(args.output_folder, unique_id)

    # Print the job data as provided by the service (this is just for cloud ml based runs)
    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    logging.info('original job data -> %s', env.get('job', {}))

    if dsbhelper.text_to_bool(args.run_training):
        run_training(args, unique_id)

    if dsbhelper.text_to_bool(args.run_eval):
        run_eval(args, unique_id)

    if dsbhelper.text_to_bool(args.run_predict):
        run_predict(args, unique_id)


# ---------------------------------
# runs the train op
# ---------------------------------

def run_training(args, unique_id):

    # Log start of training process
    logging.info('Starting run_training...')

    # Configure the trainer
    config = nuclei.NucleiConfig()
    config.display()

    # Download training data if necessary
    logging.info('Loading data...')
    local_train_data_path = 'stage1_train'
    if not os.path.exists(local_train_data_path):
        dsbhelper.autoload_data(args.zip_folder + '/stage1_train.zip', 'stage1_train.zip', local_train_data_path)

    # Set up the training dataset
    dataset_train = nuclei.NucleiDataset()

    dataset_train.load_data(local_train_data_path, mode='train', filter_ids=dsbhelper.validation_set)
    #dataset_train.load_data(local_train_data_path, mode='cluster-focus', filter_ids=dsbhelper.cluster_focus)
    dataset_train.prepare()

    # Set up validation dataset
    dataset_val = nuclei.NucleiDataset()
    dataset_val.load_data(local_train_data_path, mode='validate', filter_ids=dsbhelper.validation_set)
    dataset_val.prepare()

    # Create model in training mode
    logging.info('Loading MaskRCNN model in training mode...')
    local_model_folder = './model'
    local_weights_fname = local_model_folder + '/' + args.model_name + '.h5'
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=local_model_folder)

    # Load pretrained weights for resnet50 backbone or download last weight file and load
    logging.info('Loading pretrained weights from {}'.format(args.pretrained_weights_name))
    if args.pretrained_weights_name == 'imagenet':
        model.load_weights(dsbhelper.get_pretrained_weights(pretrained=args.pretrained_weights_name), by_name=True)

    elif args.pretrained_weights_name == 'coco':
        model.load_weights(dsbhelper.get_pretrained_weights(pretrained=args.pretrained_weights_name), by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    elif args.pretrained_weights_name == 'last':
        gcs_weights_fname = args.model_folder + '/' + args.model_name + '.h5'
        model.load_weights(dsbhelper.get_pretrained_weights(pretrained=args.pretrained_weights_name,
                                                            gcs_weights_fname=gcs_weights_fname), by_name=True)

    # Train the head branches
    logging.info('Running train op #1...')
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=config.EPOCHS, layers='heads')

    # save weights locally and copy to gcs
    model.keras_model.save_weights(local_weights_fname)
    dsbhelper.copy_file_to_gcs(local_weights_fname, args.model_folder + '/' + args.model_name + '.h5')

    # Train op #2: fine tune all layers
    logging.info('Running train op #2...')
    logging.info('Loading MaskRCNN model in training mode...')
    local_model_folder = './model'
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=local_model_folder)
    model.load_weights(local_weights_fname, by_name=True)
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE_2, epochs=config.EPOCHS_2, layers='all')

    # save weights locally and copy to gcs
    model.keras_model.save_weights(local_weights_fname)
    dsbhelper.copy_file_to_gcs(local_weights_fname, args.model_folder + '/' + args.model_name + '.h5')

    # Train op #3: further fine tune all layers
    logging.info('Running train op #3...')
    logging.info('Loading MaskRCNN model in training mode...')
    local_model_folder = './model'
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=local_model_folder)
    model.load_weights(local_weights_fname, by_name=True)
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE_3, epochs=config.EPOCHS_3, layers='all')

    # save weights locally and copy to gcs
    model.keras_model.save_weights(local_weights_fname)
    dsbhelper.copy_file_to_gcs(local_weights_fname, args.model_folder + '/' + args.model_name + '.h5')


# ---------------------------------
# runs the eval op
# ---------------------------------

def run_eval(args, unique_id):

    # define a dict for the ground truth that mirrors the return of model.detect(...)
    ground_truth = dict()
    ground_truth['rois'] = None
    ground_truth['masks'] = None
    ground_truth['class_ids'] = None
    ground_truth['scores'] = None

    # lists to store diagnositic metrics info
    ap = []                 # list to calculate mAP from, also used in csv generation
    specimen_ids = []       # list of specimen_ids for csv of ap per image

    # log start of eval process
    logging.info('Starting run_eval...')

    config = nuclei.InferenceConfig()

    # setting up datasets
    logging.info('Loading data...')
    local_train_data_path = 'stage1_train'

    # download the train data to the local instance
    if not os.path.exists(local_train_data_path):
        dsbhelper.autoload_data(args.zip_folder + '/stage1_train.zip', 'stage1_train.zip', local_train_data_path)

    # Validation dataset
    dataset_val = nuclei.NucleiDataset()
    dataset_val.load_data(local_train_data_path, mode='validate', filter_ids=dsbhelper.validation_set)
    dataset_val.prepare()

    # Recreate the model in inference mode
    logging.info('Loading MaskRCNN model in inference mode...')
    local_model_folder = './model'
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=local_model_folder)

    # load trained weight file
    logging.info('Loading trained weights from {}'.format(args.model_folder + '/' + args.model_name + '.h5'))

    gcs_weights_fname = args.model_folder + '/' + args.model_name + '.h5'
    model.load_weights(dsbhelper.get_pretrained_weights(pretrained='last',
                                                        gcs_weights_fname=gcs_weights_fname), by_name=True)

    for i, image_id in enumerate(dataset_val.image_ids):

        logging.info('Processing image #{}...'.format(i))

        # get image data (need specimen_id for filenaming below)
        image_data = dataset_val.get_info(image_id)

        fname = './output/{}_{}'.format(image_data['specimen_id'], i)

        # get the image to eval resized from original to image dims as specified in nuclei.py
        image, \
            image_meta, \
            ground_truth['class_ids'], \
            ground_truth['rois'], \
            ground_truth['masks'] = modellib.load_image_gt(dataset_val, config, image_id, use_mini_mask=False)

        # get the prediction
        p = model.detect([image], verbose=1)
        prediction = p[0]

        # rescale predictions to original image (removing mirroring if its turned on)
        original_image, \
            clean_masks, \
            clean_rois, \
            clean_class_ids, \
            clean_scores = dsbhelper.restore_original_example(dataset_val, image, prediction['masks'],
                                                              prediction['class_ids'], image_meta,
                                                              scores=prediction['scores'])

        # rescale the ground truth to original image (remove mirroring if its turned on)
        original_image, \
            ground_truth['masks'], \
            ground_truth['rois'], \
            ground_truth['class_ids'], \
            _ = dsbhelper.restore_original_example(dataset_val, image, ground_truth['masks'], ground_truth['class_ids'],
                                                   image_meta, scores=None)
        # save the images locally
        gt_fname = fname + '_gt.png'
        dsbhelper.save_instances(original_image, ground_truth['rois'], ground_truth['masks'], ground_truth['class_ids'],
                                 dataset_val.class_names, gt_fname, figsize=(16, 16))

        pd_fname = fname + '_pd.png'
        dsbhelper.save_instances(original_image, clean_rois, clean_masks, clean_class_ids,
                                 dataset_val.class_names, pd_fname, scores=clean_scores, figsize=(16, 16))

        # calculate average precision and log
        average_precision = dsbhelper.compute_ap(ground_truth['masks'], clean_masks)
        logging.info('AP = {:1.3f}'.format(average_precision))

        # store diagnostics for saving and further calcs
        ap.append(average_precision)
        specimen_ids.extend([image_data['specimen_id']])

    # store csv of average precision for each specimen
    dsbhelper.write_diagnostics(args, specimen_ids, ap, unique_id)

    # copy all of the pngs for the eval set to gcs
    dsbhelper.copy_file_pattern_to_gcs(args.output_folder, ['./output/*.png'])

    logging.info('mAP = {:1.3f}'.format(np.mean(ap)))


# ---------------------------------
# runs the predict op
# ---------------------------------

def run_predict(args, unique_id):

    # log start of eval process
    logging.info('Starting run_predict...')

    # configure the trainer
    config = nuclei.InferenceConfig()

    # setting up datasets
    logging.info('Loading data...')
    local_test_data_path = 'stage1_test'

    # download the train data to the local instance
    if not os.path.exists(local_test_data_path):
        dsbhelper.autoload_data(args.zip_folder + '/stage1_test.zip', 'stage1_test.zip', local_test_data_path)

    # Validation dataset
    dataset_test = nuclei.NucleiDataset()

    dataset_test.load_data(local_test_data_path, mode='test', filter_ids=None)
    dataset_test.prepare()

    # Recreate the model in inference mode
    logging.info('Loading MaskRCNN model in inference mode...')
    local_model_folder = './model'
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=local_model_folder)

    # load trained weight file
    logging.info('Loading trained weights from {}'.format(args.model_folder + '/' + args.model_name + '.h5'))
    gcs_weights_fname = args.model_folder + '/' + args.model_name + '.h5'
    model.load_weights(dsbhelper.get_pretrained_weights(pretrained=args.pretrained_weights_name,
                                                        gcs_weights_fname=gcs_weights_fname), by_name=True)

    # initialize lists
    instance_specimen_ids = []
    specimen_ids = []
    instance_rles = []
    rles = []

    for i, image_id in enumerate(dataset_test.image_ids):

        logging.info('Processing image #{}...'.format(i))

        # load an image
        image = dataset_test.load_image(image_id)
        image, image_meta = dsbhelper.resize_test_image(dataset_test, config, image, image_id)

        # get image data (need specimen_id below)
        image_data = dataset_test.get_info(image_id)
        fname = './output/{}_{}'.format(image_data['specimen_id'], i)

        # get the prediction
        p = model.detect([image], verbose=1)
        prediction = p[0]

        # get predict info on the original image
        original_image, \
            clean_masks, \
            clean_rois, \
            clean_class_ids, \
            clean_scores = dsbhelper.restore_original_example(dataset_test, image, prediction['masks'],
                                                              prediction['class_ids'], image_meta,
                                                              scores=prediction['scores'])

        # encode masks and get revised mask data for diagnostic images
        instance_specimen_ids, instance_rles, revised_masks = dsbhelper.encode_masks(clean_masks,
                                                                                     image_data['specimen_id'])
        rles.extend(instance_rles)
        specimen_ids.extend(instance_specimen_ids)

        # save the test images
        pd_fname = fname + '_tst.png'
        dsbhelper.save_instances(original_image, clean_rois, revised_masks, clean_class_ids,
                                 dataset_test.class_names, pd_fname, scores=clean_scores, figsize=(16, 16))

    # copy all of the pngs for the eval set to gcs
    dsbhelper.copy_file_pattern_to_gcs(args.output_folder, ['./output/*.png'])

    # write the rle out to a csv (copies to gcs within functions)
    dsbhelper.write_predictions(args, specimen_ids, rles, unique_id)


def main():
    logging.getLogger().setLevel(logging.INFO)
    run()


if __name__ == '__main__':
    main()
