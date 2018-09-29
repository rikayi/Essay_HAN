from base.base_model import BaseModel
from .HAN_utils import *
import tensorflow as tf
from tensorflow.contrib import rnn



class HANModel(BaseModel):
    def __init__(self, data_loader, config):
        super(HANModel, self).__init__(config)
        # Get the data_generators to make the joint of the inputs in the graph
        self.data_loader = data_loader
        # define some important variables
        self.x = None
        self.y = None
        self.loss = None
        self.optimizer = None
        self.train_step = None

        self.vocab_size = self.config.vocab_size
        self.embedding_size = self.config.embedding_size
        self.hidden_size = self.config.hidden_size
        self.word_ctx_size = self.config.word_ctx_size
        self.sentence_ctx_size = self.config.sentence_ctx_size
        self.batch_size=self.config.batch_size
        self.max_sentence_num = self.config.max_sent
        self.max_sentence_length = self.config.max_word


        self.build_model()
        self.init_saver()

    def build_model(self):
        """

        :return:
        """
        
        """
        Helper Variables
        """
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)
        
        """
        Inputs to the network
        """
        with tf.variable_scope('inputs'):
            self.x, self.y = self.data_loader.get_inputs()

            
        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.y)


        
        

        """
        Network Architecture
        """

        with tf.variable_scope('network'):
            with tf.name_scope('embedding'):
                embedded_weights = tf.Variable(tf.random_normal((self.vocab_size, self.embedding_size)),
                                               name='embedded_weights')
                embedded_x = tf.nn.embedding_lookup(embedded_weights, self.x)

            with tf.name_scope('sentence_vector_construction'):
                # reshape_embedded_x.shape = [batch_size*num_sentence, sentence_length, embedding_size]
                reshape_embedded_x = tf.reshape(embedded_x, [-1, self.max_sentence_length, self.embedding_size])
                # encoded_words.shape = [batch_size*num_sentence, sentence_length, hidden_size * 2]
                encoded_words = self._word_encoder(reshape_embedded_x)
                # words_attention.shape = [batch_size*num_sentence, sentence_length]
                words_attention = self._word_attention(encoded_words)
                # expand_word_attention.shape = [batch_size*num_sentence, sentence_length, 1]
                expand_word_attention = tf.expand_dims(words_attention, -1)
                # words_with_attention.shape = [batch_size*num_sentence, sentence_length, hidden_size * 2]
                words_with_attention = encoded_words * expand_word_attention
                # sentences.shape = [batch_size*num_sentence, hidden_size * 2]
                sentences = tf.reduce_sum(words_with_attention, axis=1)

            with tf.name_scope('document_vector_construction'):
                # reshape_sentences.shape = [batch_size, num_sentence, hidden_size * 2]
                reshape_sentences = tf.reshape(sentences, shape=[
                    -1, self.max_sentence_num, self.hidden_size * 2])
                # encoded_sentences.shape = [batch_size, num_sentence, hidden_size * 2]
                encoded_sentences = self._sentence_encoder(reshape_sentences)
                # sentence_attention.shape = [batch_size, num_sentence]
                sentence_attention = self._sentence_attention(encoded_sentences)

                expand_sentence_attention = tf.expand_dims(sentence_attention, -1)
                # sentences_with_attention.shape = [batch_size, num_sentence, hidden_size * 2]
                sentences_with_attention = encoded_sentences * expand_sentence_attention

                # document_vectors = [batch_size, hidden_size * 2]
                document_vectors = tf.reduce_sum(sentences_with_attention, axis=1)

            with tf.name_scope('document_prediction'):
                W_c = tf.Variable(tf.truncated_normal([self.hidden_size * 2, 1]), name='class_weights')
                b_c = tf.Variable(tf.truncated_normal([1]), name='class_biases')
                out = tf.matmul(document_vectors, W_c) + b_c
        
            with tf.variable_scope('out'):
                # self.out = [batch_size, 1]
                self.out = tf.sigmoid(out)
                tf.add_to_collection('out', self.out)

            

        """
        Some operators for the training process

        """

        with tf.variable_scope('loss'):
            self.y = tf.expand_dims(self.y, axis=1)
            self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.out)


        with tf.variable_scope('train_step'):
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)

    def init_saver(self):
        """
        initialize the tensorflow saver that will be used in saving the checkpoints.
        :return:
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)


    def _word_encoder(self, embedded_x):
        with tf.variable_scope('word_encoder'):
            word_cell_fw = rnn.GRUCell(num_units=self.hidden_size)
            word_cell_bw = rnn.GRUCell(num_units=self.hidden_size)
            # outputs shape: [batch_size, max_time, state_size]
            word_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=word_cell_fw,
                                                              cell_bw=word_cell_bw,
                                                              inputs=embedded_x,
                                                              dtype=tf.float32,
                                                              sequence_length=length(embedded_x))
            # word = [h_fw, h_bw]
            return tf.concat(word_outputs, axis=2, name='encoded_words')

    def _word_attention(self, encoded_words):
        with tf.name_scope('word_attention'):
            encoded_word_dims = self.hidden_size * 2
            # word attention layer
            word_context = tf.Variable(tf.truncated_normal([self.word_ctx_size]), name='word_context')
            W_word = tf.Variable(tf.truncated_normal(shape=[encoded_word_dims, encoded_word_dims]),
                                 name='word_context_weights')
            b_word = tf.Variable(tf.truncated_normal(shape=[encoded_word_dims]), name='word_context_bias')

            # U_{it} = tanh(W_w * h_{it} + b_w)
            U_w = tf.tanh(matrix_batch_vectors_mul(W_word, encoded_words,
                                                   [-1, self.max_sentence_length, encoded_word_dims]) + b_word,
                          name='U_w')
            word_logits = batch_vectors_vector_mul(U_w, word_context, [-1, self.max_sentence_length])
            return tf.nn.softmax(logits=word_logits)

    def _sentence_encoder(self, sentences):
        with tf.variable_scope('sentence_encoder'):
            sentence_cell_fw = rnn.GRUCell(num_units=self.hidden_size)
            sentence_cell_bw = rnn.GRUCell(num_units=self.hidden_size)
            sentence_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=sentence_cell_fw,
                                                                  cell_bw=sentence_cell_bw,
                                                                  inputs=sentences,
                                                                  dtype=tf.float32,
                                                                  sequence_length=length(sentences))
            return tf.concat(sentence_outputs, axis=2, name='encoded_sentences')

    def _sentence_attention(self, encoded_sentences):
        with tf.name_scope('sentence_attention'):
            encoded_sentence_dims = self.hidden_size * 2
            sentence_context = tf.Variable(tf.truncated_normal([encoded_sentence_dims]), name='sentence_context')
            W_sen = tf.Variable(tf.truncated_normal(shape=[encoded_sentence_dims, encoded_sentence_dims],
                                                    name='context_sentence_weights'))
            b_sen = tf.Variable(tf.truncated_normal(shape=[encoded_sentence_dims]), name='context_sentence_bias')
            U_s = tf.tanh(matrix_batch_vectors_mul(W_sen, encoded_sentences,
                                                   [-1, self.max_sentence_num, encoded_sentence_dims]) + b_sen,
                          name='U_s')
            sentence_logits = batch_vectors_vector_mul(U_s, sentence_context, [-1, self.max_sentence_num])
            return tf.nn.softmax(logits=sentence_logits)
