"""Input pipeline for the dataset.

"""
import os
import sys

sys.path.extend(['..'])

import numpy as np
import tensorflow as tf


from utils.utils import get_args
from utils.config import process_config



class DataLoader:
    def __init__(self, config):
        self.config = config

        self.train_x = np.load('../data/x_train.npy')
        self.train_y = np.load('../data/y_train.npy')
        self.test_x = np.load('../data/x_test.npy')
        self.test_y = np.load('../data/y_test.npy')
        
        # Define datasets sizes
        self.train_size = self.train_x.shape[0]
        self.test_size = self.test_x.shape[0]

        # Define number of iterations per epoch
        self.num_iterations_train = (self.train_size + self.config.batch_size - 1) // self.config.batch_size
        self.num_iterations_test  = (self.test_size  + self.config.batch_size - 1) // self.config.batch_size

        self.features_placeholder = None
        self.labels_placeholder = None

        self.dataset = None
        self.iterator = None
        self.init_iterator_op = None
        self.next_batch = None

        

        self._build_dataset_api()




    def _build_dataset_api(self):
        with tf.device('/cpu:0'):
            self.features_placeholder = tf.placeholder(tf.int32, [None, self.config.max_sent, self.config.max_word])
            self.labels_placeholder = tf.placeholder(tf.float32,[None,])

            # Create a Dataset serving batches of images and labels
            # We don't repeat for multiple epochs because we always train and evaluate for one epoch

            self.dataset = (tf.data.Dataset.from_tensor_slices(
                    (self.features_placeholder, self.labels_placeholder)
                )
                .batch(self.config.batch_size)
                .prefetch(1)  # make sure you always have one batch ready to serve
            )

            # Create reinitializable iterator from dataset
            self.iterator = self.dataset.make_initializable_iterator()

            self.iterator_init_op = self.iterator.initializer

            self.next_batch = self.iterator.get_next()

    def initialize(self, sess, mode='train'):
        if mode == 'train':
            idx = np.array(range(self.train_size))
            np.random.shuffle(idx)

            self.train_x = self.train_x[idx]
            self.train_y = self.train_y[idx]

            sess.run(self.iterator_init_op, feed_dict={self.features_placeholder: self.train_x,
                                                   self.labels_placeholder: self.train_y,
                                                   })
        
        else:
            sess.run(self.iterator_init_op, feed_dict={self.features_placeholder: self.test_x,
                                                   self.labels_placeholder: self.test_y,
                                                   })


    def get_inputs(self):
        return self.next_batch


def main(config):
    """
    Function to test from console
    :param config:
    :return:
    """
    tf.reset_default_graph()

    sess = tf.Session()


    data_loader = DataLoader(config)
    features, labels = data_loader.get_inputs()
    print('Train')

    data_loader.initialize(sess, mode='train')

    out_f, out_l = sess.run([features, labels])

    print( out_f[0])
    print( out_l)


    
    print('Test')
    data_loader.initialize(sess, mode='test')

    out_f, out_l = sess.run([features, labels])

    print( out_f[0])
    print( out_l)



if __name__ == '__main__':
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
        main(config)

    except Exception as e:
        print('Missing or invalid arguments %s' % e)
