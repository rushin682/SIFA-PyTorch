from datetime import datetime
import json
import numpy as np
import random
import os
import cv2

import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Convolution2D, DilatedConv2D, Deconvolution2D
from modules import Residual_block, Dilated_Residual_block
from models import Generator_S_T, Discriminator_T, Discriminator_AUX, Encoder, Decoder, Segmenter
from losses import cycle_consistency_loss, generator_loss, discriminator_loss, task_loss

from stats_func import *

save_interval = 300

class SIFA:
    "SIFA end-to-end network module"

    def __init__(self, config):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._source_train_pth = config['source_train_pth']
        self._target_train_pth = config['target_train_pth']
        self._source_val_pth = config['source_val_pth']
        self._target_val_pth = config['target_val_pth']
        self._output_root_dir = config['output_root_dir']
        if not os.path.isdir(self._output_root_dir):
            os.makedirs(self._output_root_dir)
        self._output_dir = os.path.join(self._output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 20
        self._pool_size = int(config['pool_size'])
        self._lambda_a = float(config['_LAMBDA_A'])
        self._lambda_b = float(config['_LAMBDA_B'])

        self._skip_conn = bool(config['skip_conn'])
        self._num_cls = int(config['num_cls'])
        self._base_lr = float(config['base_lr'])
        self._epoch = int(config['epoch'])
        self._dropout_rate = float(config['dropout_rate'])
        self._is_training = bool(config['is_training'])
        self._batch_size = int(config['batch_size'])
        self._lr_gan_decay = bool(config['lr_gan_decay'])
        self._lsgan_loss_p_scheduler = bool(config['lsgan_loss_p_scheduler'])
        self._to_restore = bool(config['to_restore'])
        self._checkpoint_dir = config['checkpoint_dir']

        self.fake_images_A = np.zeros(
            (self._pool_size, self._batch_size, model.IMG_HEIGHT, model.IMG_WIDTH, 1))
        self.fake_images_B = np.zeros(
            (self._pool_size, self._batch_size, model.IMG_HEIGHT, model.IMG_WIDTH, 1))


def train():
    # Load Dataset
    self.inputs = data_loader.load_data(self._source_train_pth, self._target_train_pth, True)
    self.inputs_val = data_loader.load_data(self._source_val_pth, self._target_val_pth, True)



def main(config_filename):

    with open(config_filename) as config_file:
        config = json.load(config_file)

    uda = SIFA(config)
    uda.train()


if __name__ == '__main__':
    main(config_filename='./config_param.json')
