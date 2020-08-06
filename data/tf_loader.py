import tensorflow as tf
import json

BATCH_SIZE = 1
domain = ''

def _decode_sample(slice_record):
    decomp_feature = {
        # image size, dimensions of 3 consecutive slices
        'dsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
        'dsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
        'dsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 3
        # label size, dimension of the middle slice
        'lsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
        'lsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
        'lsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 1
        # image slices of size [256, 256, 3]
        'data_vol': tf.io.FixedLenFeature([], tf.string),
        # label slice of size [256, 256, 1]
        'label_vol': tf.io.FixedLenFeature([], tf.string)}

    raw_size = [256, 256, 3]
    volume_size = [256, 256, 3]
    label_size = [256, 256, 1]

    parser = tf.io.parse_single_example(slice_record, features=decomp_feature)

    data_vol = tf.io.decode_raw(parser['data_vol'], tf.float32)
    data_vol = tf.reshape(data_vol, raw_size)
    data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)
    data_vol = tf.transpose(data_vol, perm=[2,0,1])
    # For converting the value range to be [-1 1] using the equation 2*[(x-x_min)/(x_max-x_min)]-1.
    # The values {-1.8, 4.4, -2.8, 3.2} need to be changed according to the statistics of specific datasets
    if 'mr' in slice_record:
        data_vol = tf.subtract(tf.multiply(tf.divide(tf.subtract(data_vol, -1.8), tf.subtract(4.4, -1.8)), 2.0), 1)
    elif 'ct' in slice_record:
        data_vol = tf.subtract(tf.multiply(tf.divide(tf.subtract(data_vol, -2.8), tf.subtract(3.2, -2.8)), 2.0), 1)


    label_vol = tf.io.decode_raw(parser['label_vol'], tf.float32)
    label_vol = tf.reshape(label_vol, raw_size)
    label_vol = tf.slice(label_vol, [0, 0, 1], label_size)
    label_vol = tf.transpose(label_vol, perm=[2,0,1])

#     batch_y = tf.squeeze(label_vol)
    # batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 5)

    return (tf.expand_dims(data_vol[1, :, :], axis=0), label_vol)


def _load_sample(path):

    return _decode_sample(path)
