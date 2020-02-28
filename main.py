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
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from layers import Convolution2D, DilatedConv2D, Deconvolution2D
from modules import Residual_block, Dilated_Residual_block
from models import SIFA, Generator_S_T, Discriminator_T, Discriminator_AUX, Discriminator_MASK, Encoder, Decoder, Segmenter
from losses import cycle_consistency_loss, generator_loss, discriminator_loss, task_loss

from stats_func import *

save_interval = 300

class UDA:
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
        self._segmentation_lr = float(config['sementation_lr'])
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


    def train(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #-------------------- SETTINGS: DATASET BUILDERS
        # Load Dataset from dataloader
        # self.inputs = data_loader.load_data(self._source_train_pth, self._target_train_pth, True)
        # self.inputs_val = data_loader.load_data(self._source_val_pth, self._target_val_pth, True)


        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        self.model = SIFA(skip_conn=self._skip_conn, is_training=self._is_training, dropout_rate=self._dropout_rate)
        self.model = nn.DataParallel(self.model).cuda()

        # Restore model to run from last checkpoint
        # if self._to_restore:



        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        self.generator_s_t_optimizer = optim.Adam(self.model.generator_s_t.parameters(), lr=self._base_lr, betas=(0.5, 0.999))
        self.generator_t_s_optimizer = optim.Adam(self.model.decoder.parameters(), lr=self._base_lr, betas=(0.5, 0.999))

        self.discriminator_s_optimizer = optim.Adam(self.model.discriminator_s.parameters(), lr=self._base_lr, betas=(0.5, 0.999))
        self.discriminator_t_optimizer = optim.Adam(self.model.discriminator_aux.parameters(), lr=self._base_lr, betas=(0.5, 0.999))
        self.discriminator_p_optimizer = optim.Adam(self.model.discriminator_mask.parameters(), lr=self._base_lr, betas=(0.5, 0.999))

        self.segmentation_optimizer = optim.Adam(self.model.encoder.parameters() + self.model.segmenter.parameters(), lr=self._segmentation_lr)

        self.adverserial_scheduler = StepLR(self.segmentation_optimizer, step_size=2, gamma=0.9, last_epoch=-1)


        # -------------------- TENSORBOARD STUFF


        # -------------------- INITIALIZATIONS BEFORE LOOP
        self.num_fake_inputs = 0
        # self.max_images = something

        self.model.train()
        for (images_s, images_t, gts_s, gts_t) in enumerate(dataloader):








def main(config_filename):

    with open(config_filename) as config_file:
        config = json.load(config_file)

    uda = UDA(config)
    uda.train()


if __name__ == '__main__':
    main(config_filename='./config_param.json')
