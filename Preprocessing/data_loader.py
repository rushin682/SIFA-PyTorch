import tensorflow as tf
import json

# with open('./config_param.json') as config_file:
#     config = json.load(config_file)

# BATCH_SIZE = int(config['batch_size'])
BATCH_SIZE = 1

def _decode_samples(image_list, shuffle=False):
    decomp_feature = {
        # image size, dimensions of 3 consecutive slices
        'dsize_dim0': tf.FixedLenFeature([], tf.int64), # 256
        'dsize_dim1': tf.FixedLenFeature([], tf.int64), # 256
        'dsize_dim2': tf.FixedLenFeature([], tf.int64), # 3
        # label size, dimension of the middle slice
        'lsize_dim0': tf.FixedLenFeature([], tf.int64), # 256
        'lsize_dim1': tf.FixedLenFeature([], tf.int64), # 256
        'lsize_dim2': tf.FixedLenFeature([], tf.int64), # 1
        # image slices of size [256, 256, 3]
        'data_vol': tf.FixedLenFeature([], tf.string),
        # label slice of size [256, 256, 1]
        'label_vol': tf.FixedLenFeature([], tf.string)}

    raw_size = [256, 256, 3]
    volume_size = [256, 256, 3]
    label_size = [256, 256, 1]

    data_queue = tf.train.string_input_producer(image_list, shuffle=shuffle)
    reader = tf.TFRecordReader()
    fid, serialized_example = reader.read(data_queue)
    parser = tf.parse_single_example(serialized_example, features=decomp_feature)

    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    data_vol = tf.reshape(data_vol, raw_size)
    data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)

    label_vol = tf.decode_raw(parser['label_vol'], tf.float32)
    label_vol = tf.reshape(label_vol, raw_size)
    label_vol = tf.slice(label_vol, [0, 0, 1], label_size)

    batch_y = tf.squeeze(label_vol)
    # batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 5)

    return tf.expand_dims(data_vol[:, :, 1], axis=2), batch_y


def _load_samples(domain_pth):

    with open(domain_pth, 'r') as fp:
        rows = fp.readlines()
    image_list = [row[:-1] for row in rows]

    data_vol, label_vol = _decode_samples(image_list, shuffle=True)

    return data_vol, label_vol


def load_data(domain_pth):

    image, gt = _load_samples(domain_pth)

    # For converting the value range to be [-1 1] using the equation 2*[(x-x_min)/(x_max-x_min)]-1.
    # The values {-1.8, 4.4, -2.8, 3.2} need to be changed according to the statistics of specific datasets
    if 'mr' in domain_pth:
        image = tf.subtract(tf.multiply(tf.div(tf.subtract(image, -1.8), tf.subtract(4.4, -1.8)), 2.0), 1)
    elif 'ct' in domain_pth:
        image = tf.subtract(tf.multiply(tf.div(tf.subtract(image, -2.8), tf.subtract(3.2, -2.8)), 2.0), 1)

    if 'ct' in domain_pth:
        image = tf.subtract(tf.multiply(tf.div(tf.subtract(image, -2.8), tf.subtract(3.2, -2.8)), 2.0), 1)
    elif 'mr' in domain_pth:
        image = tf.subtract(tf.multiply(tf.div(tf.subtract(image, -1.8), tf.subtract(4.4, -1.8)), 2.0), 1)


    # Batch
    # Do we need this?
    images, gt = tf.train.batch([image, gt], batch_size=BATCH_SIZE, num_threads=1, capacity=10000)

    return images, gt
