from datetime import datetime
import json
import numpy as np
import os
import random

import tensorflow as tf

import tf_loader

import csv

source_train_pth = './data/datalist/training_A.txt'
target_train_pth = './data/datalist/training_B.txt'
source_val_pth = './data/datalist/validation_A.txt'
target_val_pth = './data/datalist/validation_B.txt'

num_cls = 5

class tf2nii:
    """The tfrecord to nii.gz converter"""

    def __init__(self, output_dir):

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._source_train_pth = source_train_pth
        self._target_train_pth = target_train_pth
        self._source_val_pth = source_val_pth
        self._target_val_pth = target_val_pth
        self._num_cls = num_cls

        self._output_dir = os.path.join(output_dir, current_time)

        # Load Dataset from the dataset folder
        self.source_inputs = tf_loader.load_data(self._source_train_pth)
        self.target_inputs = tf_loader.load_data(self._target_train_pth)

        self.source_inputs_val = tf_loader.load_data(self._source_val_pth)

        # Save Images Counter
        self.count = 0

        # Initializing the global variables
        init = (tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.local_variables_initializer())

        with open(self._source_train_pth, 'r') as fp:
            rows_s = fp.readlines()
        with open(self._target_train_pth, 'r') as fp:
            rows_t = fp.readlines()
        with open(self._source_val_pth, 'r') as fp:
            rows_s_val = fp.readlines()
        with open(self._target_val_pth, 'r') as fp:
            rows_t_val = fp.readlines()

        num_source_samples = len(rows_s)
        print(len(rows_s))
        # num_target_samples = len(rows_t)
        # num_source_val_samples = len(rows_s_val)
        # num_target_val_samples = len(rows_t_val)

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto()) as sess:
            # sess.run(init)

            if not os.path.exists(os.path.join("data", self._output_dir)):
                os.makedirs(os.path.join("data", self._output_dir))

            # Save Source Train files
            self.looper(sess, self.source_inputs, num_source_samples, export_dir="mr_train", export_csv="mr_train.csv")

            # # Save Source Val files
            # self.looper(sess, self.source_inputs_val, num_source_val_samples, export_dir="mr_val", export_csv="mr_val.csv")
            #
            # # Save Target Train files
            # self.looper(sess, self.target_inputs, num_target_samples, export_dir="ct_train", export_csv="ct_train.csv")
            #
            # # Save Target Val files
            # self.looper(sess, self.target_inputs_val, num_target_val_samples, export_dir="ct_val", export_csv="ct_val.csv")



    def looper(self, sess, domain_mode, num_slices, export_dir=None, export_csv=None):
        self.count = 0
        self.export_dir = os.path.join("data", self._output_dir, export_dir)
        self.export_csv = os.path.join("data", self._output_dir, "datalist", export_csv)
        for i in range(num_slices):

            images, gts = sess.run(domain_mode)
            # images = sess.run(images)
            # gts = sess.run(gts)
            self.save_file(images, gts)


    def save_file(self, image, label):
        # save to the disk
        self.count += 1

        image_name = "coronal_slice_"+"{:04d}".format(self.count)
        label_name = "coronal_slice_label"+"{:04d}".format(self.count)

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
            os.makedirs(os.path.join(self.export_dir, "slices"))
            os.makedirs(os.path.join(self.export_dir, "labels"))

        with open(self.export_csv, 'a', newline='') as csvfile:
            wrt = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            wrt.writerow([image_name, label_name])

        nii_image = nib.Nifti1Image(image, np.diag([1,1,1,1]))
        nii_label = nib.Nifti1Image(label, np.diag([1,1,1,1]))

        nib.save(nii_image, os.path.join(export_dir, "slices", image_name+".nii.gz"))
        nib.save(nii_label, os.path.join(export_dir, "labels", label_name+".nii.gz"))




def main(output_dir):
    dataloader = tf2nii(output_dir)


if __name__ == '__main__':
    output_dir = "ct_mr_dataset"
    main(output_dir)
