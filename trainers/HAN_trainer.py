from base.base_train import BaseTrain
from tqdm import tqdm
import os

import tensorflow as tf
import numpy as np

from utils.metrics import AverageMeter
from utils.kappa import quadratic_weighted_kappa


class HANTrainer(BaseTrain):
    def __init__(self, sess, model, config, logger, data_loader):
        """
        Constructing trainer based on the Base Train..
        Here is the pipeline of constructing
        - Assign sess, model, config, logger, data_generators(if_specified)
        - Initialize all variables
        - Load the latest checkpoint
        - Create the summarizer
        - Get the nodes we will need to run it from the graph
        :param sess:
        :param model:
        :param config:
        :param logger:
        :param data_loader:
        """

        super(HANTrainer, self).__init__(sess, model, config, logger, data_loader)

        # load the model from the latest checkpoint
        self.model.load(self.sess)

        # Summarizer
        self.summarizer = logger

        self.x, self.y = tf.get_collection('inputs')
        self.train_op, self.loss_node = tf.get_collection('train')
        self.out = tf.get_collection('out')
    
    def train(self):
        """
        This is the main loop of training
        Looping on the epochs
        :return:
        """
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.test(cur_epoch)

    def train_epoch(self, epoch=None):
        """
        Train one epoch
        :param epoch: cur epoch number
        :return:
        """
        # initialize dataset
        self.data_loader.initialize(self.sess, mode='train')

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="epoch-{}-".format(epoch))

        loss_per_epoch = AverageMeter()
        kappa_per_epoch = AverageMeter()

        # Iterate over batches
        for cur_it in tt:
            # One step on the current batch
            loss, kappa = self.step()
            # update metrics returned from train_step func
            loss_per_epoch.update(loss)
            kappa_per_epoch.update(kappa)

        self.sess.run(self.model.global_epoch_inc)

        # summarize
        summaries_dict = {'train/loss_per_epoch': loss_per_epoch.val,
                          'train/kappa_per_epoch': kappa_per_epoch.val}
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        self.model.save(self.sess)
        
        print("""
Epoch-{}  loss:{:.4f} -- kappa:{:.4f}
        """.format(epoch, loss_per_epoch.val, kappa_per_epoch.val))

        tt.close()

    def step(self):
        """
        Run the session of step in tensorflow
        also get the loss & kappa of that minibatch.
        :return: (loss, kappa) tuple of some metrics to be used in summaries
        """
        _, loss, y, predictions = self.sess.run([self.train_op, self.loss_node, self.y, self.out])

        #rescaling to original values
        predictions = np.squeeze(np.array(predictions))
        y = np.around(y*(self.config.max_rating-self.config.min_rating)+self.config.min_rating)
        predictions = np.around(predictions*(self.config.max_rating-self.config.min_rating)+self.config.min_rating)

        kappa = quadratic_weighted_kappa(y, predictions, self.config.min_rating, self.config.max_rating)
        return loss, kappa
    
    def test(self, epoch):
        # initialize dataset
        self.data_loader.initialize(self.sess, mode='test')

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_test), total=self.data_loader.num_iterations_test,
                  desc="Val-{}-".format(epoch))

        loss_per_epoch = AverageMeter()
        kappa_per_epoch = AverageMeter()
        # Iterate over batches
        for cur_it in tt:
            # One step on the current batch
            loss, kappa = self.step()
            # update metrics returned from step func
            loss_per_epoch.update(loss)
            kappa_per_epoch.update(kappa)

            
        # summarize
        summaries_dict = {'test/loss_per_epoch': loss_per_epoch.val,
                          'test/kappa_per_epoch': kappa_per_epoch.val}
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)
        
        print("""
Epoch-{}  loss:{:.4f} -- kappa:{:.4f}
        """.format(epoch, loss_per_epoch.val, kappa_per_epoch.val))

        tt.close()