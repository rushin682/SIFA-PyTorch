import tensorflow as tf
import data_loader

import numpy as np
import os
import cv2


def save_images(image_s, image_t):

    images_dir = "images_dir"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    num_imgs_to_save = 1
    names = ['inputA_', 'inputB_']

    for i in range(0, num_imgs_to_save):

        tensors = [image_s[i], image_t[i]]

        for name, tensor in zip(names, tensors):
            image_name = name + "_" + str(i) + ".jpg"
            cv2.imwrite(os.path.join(images_dir, image_name), ((tensor[0] + 1) * 127.5).numpy().astype(np.uint8).squeeze())


source_train_pth = "data/datalist/mr_train_list"
target_train_pth = "data/datalist/ct_train_list"

inputs = data_loader.load_data(source_train_pth, target_train_pth, True)

image_s, image_t, gt_s, gt_t = inputs
print("bleh")
print(image_s.shape, image_t.shape)

save_images(image_s[1], image_t[1])
