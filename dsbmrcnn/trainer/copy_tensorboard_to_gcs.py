import dsbhelper
import argparse
import logging
import os
import time


def copy_events(args):

    start_time = time.time()

    while True:

        # get the newest folder
        folder_list = [os.path.join(args.model_folder, d) for d in os.listdir(args.model_folder)
                       if os.path.isdir(os.path.join(args.model_folder, d))]
        tensorboard_folder = max(folder_list, key=os.path.getmtime)

        # copy the log file to gcs
        dsbhelper.copy_file_pattern_to_gcs(args.gcs_log_folder, [tensorboard_folder + '/events.*'])

        # wait 60 seconds
        logging.info('Waiting {} secs for next update...'.format(args.wait_time))
        time.sleep(args.wait_time - ((time.time() - start_time) % args.wait_time))


def run(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_folder', dest='model_folder', default='./model',
                        help='Folder with training images and masks')

    parser.add_argument('--gcs_log_folder', dest='gcs_log_folder', default='./tflogs',
                        help='Folder with training images and masks')

    parser.add_argument('--wait_time', dest='wait_time', type=float, default=60.0,
                        help='Seconds to wait between log file updates to gcs')


    # get the command line arguments
    args, _ = parser.parse_known_args(argv)

    # make train data
    copy_events(args)


def main():
    logging.getLogger().setLevel(logging.INFO)
    run()

if __name__ == '__main__':
    main()
