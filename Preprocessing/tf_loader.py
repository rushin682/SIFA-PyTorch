import tensorflow as tf
import json

# with open('./config_param.json') as config_file:
#     config = json.load(config_file)

# BATCH_SIZE = int(config['batch_size'])
BATCH_SIZE = 1
domain = ''

def _decode_samples(slice_record):
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
    if 'mr' in domain:
        data_vol = tf.subtract(tf.multiply(tf.divide(tf.subtract(data_vol, -1.8), tf.subtract(4.4, -1.8)), 2.0), 1)
    elif 'ct' in domain:
        data_vol = tf.subtract(tf.multiply(tf.divide(tf.subtract(data_vol, -2.8), tf.subtract(3.2, -2.8)), 2.0), 1)
    
    
    label_vol = tf.io.decode_raw(parser['label_vol'], tf.float32)
    label_vol = tf.reshape(label_vol, raw_size)
    label_vol = tf.slice(label_vol, [0, 0, 1], label_size)
    label_vol = tf.transpose(label_vol, perm=[2,0,1])

#     batch_y = tf.squeeze(label_vol)
    # batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 5)

    return (tf.expand_dims(data_vol[1, :, :], axis=0), label_vol)


def _load_samples(domain_pth):

    with open(domain_pth, 'r') as fp:
        rows = fp.readlines()
    image_list = [row[:-1] for row in rows]
    
    data_queue = tf.data.TFRecordDataset(image_list)

    coronal_slices = []
    for elem in iter(data_queue):
        each_slice = _decode_samples(elem)
        coronal_slices.append(each_slice)
    
    print(len(coronal_slices))
    print(set(lab.numpy().shape for inp,lab in coronal_slices))

    return coronal_slices


def load_data(domain_pth):
    
    domain = domain_pth

    coronal_slices = _load_samples(domain_pth)

    # Batch
    # Do we need this?
    # images, gt = tf.train.batch([image, gt], batch_size=BATCH_SIZE, num_threads=1, capacity=10000)

    return coronal_slices
