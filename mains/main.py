import sys
import os



sys.path.extend(['..'])

import tensorflow as tf

from data_generators.generator_dataset import DataLoader
from models.HAN_model import HANModel
from trainers.HAN_trainer import HANTrainer

from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import DefinedSummarizer
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # Create tensorflow session and limit gpu usage
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth=True
    sess = tf.Session(config=conf)

    # create your data generator
    data_loader = DataLoader(config)

    # create instance of the model you want
    model = HANModel(data_loader, config)


    # create tensorboard logger
    logger = DefinedSummarizer(sess, summary_dir=config.summary_dir,
                               scalar_tags=['train/loss_per_epoch',
                                            'train/kappa_per_epoch',
                                            'test/loss_per_epoch',
                                            'test/kappa_per_epoch'])
                               

    # create trainer and path all previous components to it
    trainer = HANTrainer(sess, model, config, logger, data_loader)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
