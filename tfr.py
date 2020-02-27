import torch
from tfrecord.torch.dataset import MultiTFRecordDataset
import tensorflow as tf

tfrecord_pattern = "mr_train_tfs/{}.tfrecords"
index_pattern = "mr_train_list.txt"

description = {# image size, dimensions of 3 consecutive slices
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

dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, description)
loader = torch.utils.data.DataLoader(dataset, batch_size=8)

data = next(iter(loader))
print(data)
