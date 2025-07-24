import tensorflow as tf
from FaceAging import FaceAging
from os import environ
import argparse
import os
import shutil

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='CAAE')
parser.add_argument('--is_train', type=str2bool, default=True)
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--dataset', type=str, default='UTKFace', help='training dataset name that is  stored in ./data')
parser.add_argument('--savedir', type=str, default='./FaceAging_CAAE_save', help='dir for saving checkpoints and results')
parser.add_argument('--testdir', type=str, default='None', help='dir of testing images')
parser.add_argument('--use_trained_model', type=str2bool, default=True, help='whether to train from an existing model')
parser.add_argument('--use_init_model', type=str2bool, default=True, help='whether train from the init model if cannot find an existing model')
FLAGS = parser.parse_args()

def main():
    import pprint
    pprint.pprint(vars(FLAGS))

    # Logic to clear the save directory for a fresh training run
    if FLAGS.is_train and not FLAGS.use_trained_model:
        if os.path.exists(FLAGS.savedir):
            print(f"Starting a fresh training run. Deleting old save directory: {FLAGS.savedir}")
            shutil.rmtree(FLAGS.savedir)

    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    model = FaceAging(
        is_training=FLAGS.is_train,
        save_dir=FLAGS.savedir,
        dataset_name=FLAGS.dataset
    )
    
    if FLAGS.is_train:
        print('\n\tTraining Mode')
        model.train(
            num_epochs=FLAGS.epoch,
            use_trained_model=FLAGS.use_trained_model,
            use_init_model=FLAGS.use_init_model
        )
    else:
        print('\n\tTesting Mode')
        model.custom_test(
            testing_samples_dir=FLAGS.testdir + '/*.jpg'
        )

if __name__ == '__main__':
    main()
