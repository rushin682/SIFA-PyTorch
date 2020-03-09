import time
from datetime import datetime
import json
import numpy as np
import random
import os
import cv2
import csv
from math import floor


import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch._six import int_classes as _int_classes
from torch.utils.data import DataLoader

import nibabel as nib

from layers import Convolution2D, DilatedConv2D, Deconvolution2D
from modules import Residual_block, Dilated_Residual_block
import model
from model import SIFA_generators, SIFA_discriminators, Generator_S_T, Discriminator_T, Discriminator_AUX, Discriminator_MASK, Encoder, Decoder, Segmenter
import losses

from data_loader import Two_idx_BatchSampler, Two_idx_RandomSampler
from data_loader import CT_MR_Dataset

from stats_func import *

save_interval = 300

class UDA:
    "SIFA end-to-end network module"

    def __init__(self, config):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._source_train_pth = config['source_train_path']
        self._target_train_pth = config['target_train_path']
        self._source_val_pth = config['source_val_path']
        self._target_val_pth = config['target_val_path']
        self._output_root_dir = config['output_root_dir']
        if not os.path.isdir(self._output_root_dir):
            os.makedirs(self._output_root_dir)
        self._output_dir = os.path.join(self._output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 20
        self._pool_size = int(config['pool_size'])
        self._lambda_s = float(config['_LAMBDA_S'])
        self._lambda_t = float(config['_LAMBDA_T'])

        self._skip_conn = bool(config['skip_conn'])
        self._num_cls = int(config['num_cls'])
        self._base_lr = float(config['base_lr'])
        self._segmentation_lr = float(config['segmentation_lr'])
        self._num_epoch = int(config['epoch'])
        self._dropout_rate = float(config['dropout_rate'])
        self._is_training = bool(config['is_training'])
        self._batch_size = int(config['batch_size'])
        self._lr_gan_decay = bool(config['lr_gan_decay'])
        self._lsgan_loss_p_scheduler = bool(config['lsgan_loss_p_scheduler'])
        self._to_restore = bool(config['to_restore'])
        self._checkpoint_dir = config['checkpoint_dir']

        self.nGPU = config['ngpu']

        self.fake_images_s = np.zeros(
            (self._pool_size, self._batch_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH))
        self.fake_images_t = np.zeros(
            (self._pool_size, self._batch_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH))

    # '''
    # def save_images(self, epoch):
    #     if not os.path.exists(self._images_dir)
    #         os.makedirs(self._images_dir)
    #
    #
    #     names = ['inputS_', 'inputT_', 'fakeS_', 'fakeT_', 'cycS_', 'cycT_']
    #
    #     for i in range(0, self._num_imgs_to_save):
    #         # print("Saving image {}/{}".format(i, self._num_imgs_to_save))
    # '''
    #
    def fake_image_pool(self, num_fakes, fake, fake_pool):
        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake.detach()
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake.detach()
                return temp
            else:
                return fake

    def one_hot(self, gt):
        C = self._num_cls
        gt_extended=gt.clone().type(torch.LongTensor)
        # gt_extended.unsqueeze_(1) # convert to Nx1xHxW
        print("Now", gt_extended.shape)
        one_hot = torch.FloatTensor(gt_extended.size(0), C, gt_extended.size(2), gt_extended.size(3)).zero_()
        one_hot.scatter_(1, gt_extended, 1)
        return one_hot

    def train(self):

        device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

        #-------------------- SETTINGS: DATASET BUILDERS
        # Load Dataset from dataloader

        augmentations = {'rotation_angle': 15,
                         'shift_range': [0.3,0.3],
                         'shear_range': 0.1,
                         'zoom_range': 1.3 }

        train_dataset = CT_MR_Dataset(self._source_train_pth, self._target_train_pth, augment_param=augmentations)
        val_dataset = CT_MR_Dataset(self._source_val_pth, self._target_val_pth, augment_param=augmentations)

        # Custom Samplers
        two_idx_train_sampler = Two_idx_RandomSampler(train_dataset)
        two_idx_train_batch_sampler = Two_idx_BatchSampler(two_idx_train_sampler, batch_size=self._batch_size, drop_last=False)
        two_idx_val_sampler = Two_idx_RandomSampler(val_dataset)
        two_idx_val_batch_sampler = Two_idx_BatchSampler(two_idx_val_sampler, batch_size=1, drop_last=False)

        # Dataloaders
        self.dataloader_train = DataLoader(train_dataset, batch_sampler=two_idx_train_batch_sampler)
        self.dataloader_val = DataLoader(val_dataset, batch_sampler=two_idx_val_batch_sampler)


        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        self.model_generators = SIFA_generators(skip_conn=self._skip_conn, is_training=self._is_training, dropout_rate=self._dropout_rate)
        self.model_generators = self.model_generators.to(device)

        self.model_discriminators = SIFA_discriminators()
        self.model_discriminators = self.model_discriminators.to(device)

        if (device.type == 'cuda') and (self.ngpu > 1):
            self.model_generators = nn.DataParallel(self.model_generators, list(range(self.ngpu)))
            self.model_discriminators = nn.DataParallel(self.model_discriminators, list(range(self.ngpu)))


        # Restore model to run from last checkpoint
        # if self._to_restore:



        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        curr_lr = self._base_lr
        curr_seg_lr = self._segmentation_lr

        self.generator_s_t_optimizer = optim.Adam(self.model_generators.generator_s_t.parameters(), lr=curr_lr, betas=(0.5, 0.999))
        self.generator_t_s_optimizer = optim.Adam(self.model_generators.decoder.parameters(), lr=curr_lr, betas=(0.5, 0.999))

        self.discriminator_s_optimizer = optim.Adam(self.model_discriminators.discriminator_aux.parameters(), lr=curr_lr, betas=(0.5, 0.999))
        self.discriminator_t_optimizer = optim.Adam(self.model_discriminators.discriminator_t.parameters(), lr=curr_lr, betas=(0.5, 0.999))
        self.discriminator_p_optimizer = optim.Adam(self.model_discriminators.discriminator_mask.parameters(), lr=curr_lr, betas=(0.5, 0.999))

        self.segmentation_optimizer = optim.Adam(list(self.model_generators.encoder.parameters())
                                                 + list(self.model_generators.segmenter.parameters()),
                                                 lr=curr_seg_lr, weight_decay=0.0001)


        self.adverserial_scheduler = StepLR(self.segmentation_optimizer, step_size=2, gamma=0.9, last_epoch=-1)


        # -------------------- TENSORBOARD STUFF


        # -------------------- INITIALIZATIONS BEFORE LOOP
        self.num_fake_inputs = 0
        # self.max_images = something

        self.model_generators.train()
        self.model_discriminators.train()

        save_count = -1

        for epoch in range(0, self._num_epoch):

            print("In the epoch", epoch)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            if self._lr_gan_decay:
                if epoch < (self._num_epoch/2):
                    curr_lr = self._base_lr
                else:
                    curr_lr = self._base_lr - self._base_lr * (epoch - self._num_epoch/2) / (self._num_epoch/2)
            else:
                    curr_lr = self._base_lr

            if self._lsgan_loss_p_scheduler:
                    if epoch < 5:
                        lsgan_loss_p_weight_value = 0.0
                    elif epoch < 7:
                        lsgan_loss_p_weight_value = 0.1 * (epoch - 4.0) / (7.0 - 4.0)
                    else:
                        lsgan_loss_p_weight_value = 0.1
            else:
                lsgan_loss_p_weight_value = 0.1

            # self.save_images(epoch)

            for idx, batch_sample in enumerate(self.dataloader_train):

                save_count += 1
                print("Processing batch {}".format(idx))

                images_s, gts_s = batch_sample['image_source'], batch_sample['gt_source']
                images_t, gts_t = batch_sample['image_target'], batch_sample['gt_target']

                # Executing each network with the current resources
                images_s = images_s.float().to(device)
                images_t = images_t.float().to(device)

                print("gt shapes before: ", gts_s.shape, gts_t.shape)

                gts_s = self.one_hot(gts_s.float().to(device))
                gts_t = self.one_hot(gts_t.float().to(device))

                print("GT shapes after one_hot: ", gts_s.shape, gts_t.shape)

                generated_images = self.model_generators(inputs = {"images_s": images_s, "images_t": images_t})

                # Adding all the synthesized images | fake images to a list for discriminator networks
                generated_images["fake_pool_t"] = self.fake_image_pool(self.num_fake_inputs, generated_images["fake_images_t"], self.fake_images_t)
                generated_images["fake_pool_s"] = self.fake_image_pool(self.num_fake_inputs, generated_images["fake_images_s"], self.fake_images_s)
                self.num_fake_inputs += 1

                discriminator_results = self.model_discriminators(generated_images)

                # ----------Optimizing the Generator_S_T Network-----------

                # Set Zero Gradients
                self.model_generators.zero_grad()
                self.model_discriminators.zero_grad()

                # Get Loss specific to generator_s_t and propogate backwards
                cycle_consistency_loss_s = \
                    self._lambda_s * losses.cycle_consistency_loss(
                        real_images=images_s,
                        generated_images=generated_images["cycle_images_s"]
                )

                cycle_consistency_loss_t = \
                    self._lambda_t * losses.cycle_consistency_loss(
                        real_images=images_t,
                        generated_images=generated_images["cycle_images_t"]
                )

                lsgan_loss_t = losses.generator_loss(discriminator_results["prob_fake_t_is_real"])

                g1_generator_s_t_loss = cycle_consistency_loss_s + lsgan_loss_t
                g1_generator_s_t_loss.backward(retain_graph=True)

                g2_generator_s_t_loss = cycle_consistency_loss_t
                g2_generator_s_t_loss.backward(retain_graph=True)
                self.generator_s_t_optimizer.step()
                # Step optimizer



                # ----------Optimizing the Discriminator_T Network-----------

                # Set Zero Gradients
                self.model_generators.zero_grad()
                self.model_discriminators.zero_grad()

                # Get Loss specific to Discriminator_T and propogate backwards

                discriminator_t_loss_real, discriminator_t_loss_fake = losses.discriminator_loss(prob_real_is_real=discriminator_results["prob_real_t_is_real"],
                                                                 prob_fake_is_real=discriminator_results["prob_fake_pool_t_is_real"])

                discriminator_t_loss_real.backward()
                discriminator_t_loss_fake.backward()

                # Step optimizer
                self.discriminator_t_optimizer.step()




                # ----------Optimizing the Segmentation Network i.e E + C-----------

                #  Set Zero Gradients
                self.model_generators.zero_grad()
                self.model_discriminators.zero_grad()

                # Get Loss specific to Segmentation and propogate backwards

                ce_loss_t, dice_loss_t = losses.task_loss(generated_images["pred_mask_fake_t"], gts_s)

                lsgan_loss_s = losses.generator_loss(discriminator_results["prob_fake_s_is_real"])
                g1_generator_t_s_loss = cycle_consistency_loss_s
                g2_generator_t_s_loss = cycle_consistency_loss_t + lsgan_loss_s

                g1_segmentation_t_loss = ce_loss_t + dice_loss_t + 0.1*g1_generator_t_s_loss
                g1_segmentation_t_loss.backward(retain_graph=True)

                g2_segmentation_t_loss = 0.1*g2_generator_t_s_loss + lsgan_loss_p_weight_value*lsgan_loss_p + 0.1*lsgan_loss_a_aux
                g2_segmentation_t_loss.backward(retain_graph=True)

                # Step optimizer
                self.segmentation_optimizer.step()



                # ----------Optimizing the Generator_T_S Network i.e Decoder-----------

                #  Set Zero Gradients
                self.model_generators.zero_grad()
                self.model_discriminators.zero_grad()

                # Get Loss specific to Decoder and propogate backwards
                lsgan_loss_s = losses.generator_loss(discriminator_results["prob_fake_s_is_real"])
                g1_generator_t_s_loss = cycle_consistency_loss_s
                g2_generator_t_s_loss = cycle_consistency_loss_t + lsgan_loss_s

                g1_generator_t_s_loss.backward(retain_graph=True)
                g2_generator_t_s_loss.backward(retain_graph=True)

                # Step optimizer
                self.generator_t_s_optimizer.step()


                # ----------Optimizing the Discriminator_S Network i.e Discriminator_AUX-----------

                #  Set Zero Gradients
                self.model_generators.zero_grad()
                self.model_discriminators.zero_grad()

                # Get Loss specific to Decoder and propogate backwards
                discriminator_s_loss_real, discriminator_s_loss_fake = losses.discriminator_loss(prob_real_is_real=discriminator_results["prob_real_s_is_real"],
                                                                 prob_fake_is_real=discriminator_results["prob_fake_pool_s_is_real"])

                discriminator_s_aux_loss_real, discriminator_s_aux_loss_fake = losses.discriminator_loss(prob_real_is_real=discriminator_results["prob_cycle_s_aux_is_real"],
                                                                 prob_fake_is_real=discriminator_results["prob_fake_pool_s_aux_is_real"])

                pool_discriminator_s_loss = discriminator_s_loss_fake + discriminator_s_aux_loss_fake

                discriminator_s_loss_real.backward()
                discriminator_s_aux_loss_real.backward()
                pool_discriminator_s_loss.backward()

                # Step optimizer
                self.discriminator_s_optimizer.step()


                # ----------Optimizing the discriminator pred Network i.e Discriminator_MASK-----------

                #  Set Zero Gradients
                self.model_generators.zero_grad()
                self.model_discriminators.zero_grad()

                # Get Loss specific to discriminator_mask and propogate backwards
                discriminator_p_loss_real, discriminator_p_loss_fake = losses.discriminator_loss(prob_real_is_real=discriminator_results["prob_pred_mask_fake_t_is_real"],
                                                                 prob_fake_is_real=discriminator_results["prob_pred_mask_t_is_real"])

                discriminator_p_loss_real.backward()
                discriminator_p_loss_fake.backward()

                # Step optimizer
                self.discriminator_p_optimizer.step()


                self.num_fake_inputs += 1





            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            self.adverserial_scheduler.step()



def main(config_filename):

    with open(config_filename) as config_file:
        config = json.load(config_file)

    uda = UDA(config)
    uda.train()


if __name__ == '__main__':
    main(config_filename='./config.json')
