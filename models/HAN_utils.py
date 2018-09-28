import tensorflow as tf


def length(sequences):
    """
    :param sequences: shape=[batch_size, max_time_step, embedding_size]
    :return: shape=[batch] whose value is actual length of each sequence in sequences
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)


def matrix_batch_vectors_mul(mat, batch_vectors, shape_after_mul):
    """
    :param mat: [N x N] 
    :param batch_vectors: [K x M x N] 
    :param shape_after_mul: [K x M x N]
    :return: new batch vectors: [K x M x N]
    """
    vectors = tf.reshape(batch_vectors, [-1, batch_vectors.shape[-1].value])
    res = tf.matmul(mat, vectors, transpose_a=False, transpose_b=True)
    return tf.reshape(tf.transpose(res), shape_after_mul)


def batch_vectors_vector_mul(batch_vectors, vector, shape_after_mul):
    """
    :param batch_vectors: [K x M x N]
    :param vector: [N]
    :param shape_after_mul: [K x M]
    :return: [K x M]
    """
    expand_vec = tf.expand_dims(vector, -1)
    mat_vec = tf.reshape(batch_vectors, [-1, batch_vectors.get_shape()[-1].value])
    res = tf.matmul(mat_vec, expand_vec)
    return tf.reshape(res, shape_after_mul)