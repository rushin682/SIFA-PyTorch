import time
from datetime import datetime
import json
import numpy as np
import random
import os
import cv2
import csv
from math import floor, ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
        self.print_freq = config['print_frequency']
        self.save_epoch_freq = config['save_epoch_freq']

        self.ngpu = config['ngpu']

        # (pool_size, batch_size=8, channel=1, h, w)
        self.fake_images_s = torch.zeros(
            (self._pool_size, self._batch_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH))
        self.fake_images_t = torch.zeros(
            (self._pool_size, self._batch_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH))

        # (pool_size, batch_size=1, channel=1, h, w)
        self.fake_val_images_s = torch.zeros(
            (self._pool_size, 1, 1, model.IMG_HEIGHT, model.IMG_WIDTH))
        self.fake_val_images_t = torch.zeros(
            (self._pool_size, 1, 1, model.IMG_HEIGHT, model.IMG_WIDTH))

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


    def train(self):

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
        two_idx_train_batch_sampler = Two_idx_BatchSampler(two_idx_train_sampler, batch_size=self._batch_size, drop_last=True)
        two_idx_val_sampler = Two_idx_RandomSampler(val_dataset)
        two_idx_val_batch_sampler = Two_idx_BatchSampler(two_idx_val_sampler, batch_size=1, drop_last=True)

        # Dataloaders
        self.dataloader_train = DataLoader(train_dataset, batch_sampler=two_idx_train_batch_sampler, num_workers=4)
        self.dataloader_val = DataLoader(val_dataset, batch_sampler=two_idx_val_batch_sampler, num_workers=4)


        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        self.model_generators = SIFA_generators(skip_conn=self._skip_conn, is_training=self._is_training, dropout_rate=self._dropout_rate)
        self.model_discriminators = SIFA_discriminators()

        device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

        if (device.type == 'cuda') and (self.ngpu > 1):
            self.model_generators = nn.DataParallel(self.model_generators, list(range(self.ngpu)))
            self.model_discriminators = nn.DataParallel(self.model_discriminators, list(range(self.ngpu)))

        self.model_generators = self.model_generators.to(device)
        self.model_discriminators = self.model_discriminators.to(device)

        # Restore model to run from last checkpoint
        # if self._to_restore:



        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        curr_lr = self._base_lr
        curr_seg_lr = self._segmentation_lr

        self.generator_s_t_optimizer = optim.Adam(self.model_generators.module.generator_s_t.parameters(), lr=curr_lr, betas=(0.5, 0.999))
        self.generator_t_s_optimizer = optim.Adam(self.model_generators.module.decoder.parameters(), lr=curr_lr, betas=(0.5, 0.999))

        self.discriminator_s_optimizer = optim.Adam(self.model_discriminators.module.discriminator_aux.parameters(), lr=curr_lr, betas=(0.5, 0.999))
        self.discriminator_t_optimizer = optim.Adam(self.model_discriminators.module.discriminator_t.parameters(), lr=curr_lr, betas=(0.5, 0.999))
        self.discriminator_p_optimizer = optim.Adam(self.model_discriminators.module.discriminator_mask.parameters(), lr=curr_lr, betas=(0.5, 0.999))

        self.segmentation_optimizer = optim.Adam(list(self.model_generators.module.encoder.parameters())
                                                 + list(self.model_generators.module.segmenter.parameters()),
                                                 lr=curr_seg_lr, weight_decay=0.0001)


        self.adverserial_scheduler = StepLR(self.segmentation_optimizer, step_size=2, gamma=0.9, last_epoch=-1)

        # -------------------- TENSORBOARD STUFF
        writer = SummaryWriter(comment="BATCH_08")
        all_losses = {}

        # -------------------- Checkpoint Directory
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        # -------------------- INITIALIZATIONS BEFORE LOOP
        self.num_fake_inputs = 0
        self.num_fake_val_inputs = 0
        # self.max_images = something

        save_count = -1
        total_iter = 0
        total_val_iter = 0

        num_iter_per_epoch = ceil(len(train_dataset) / self._batch_size)
        num_val_iter_per_epoch = ceil(len(val_dataset))

        for epoch in range(0, self._num_epoch):
            epoch_iter = 0
            epoch_val_iter = 0

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
            # Train Loop
            self.model_generators.train()
            self.model_discriminators.train()
            
            for idx, batch_sample in enumerate(self.dataloader_train):

                epoch_iter += 1
                total_iter += 1

                save_count += 1

                # print("Processing batch {}".format(idx))

                images_s, gts_s = batch_sample['image_source'], batch_sample['gt_source']
                images_t, gts_t = batch_sample['image_target'], batch_sample['gt_target']

                # Executing each network with the current resources
                images_s = images_s.float().to(device)
                images_t = images_t.float().to(device)
                gts_s = gts_s.to(device)
                gts_t = gts_t.to(device)
                generated_images = self.model_generators(inputs = {"images_s": images_s, "images_t": images_t})

                # Adding all the synthesized images | fake images to a list for discriminator networks
                generated_images["fake_pool_t"] = self.fake_image_pool(self.num_fake_inputs, generated_images["fake_images_t"], self.fake_images_t)
                generated_images["fake_pool_s"] = self.fake_image_pool(self.num_fake_inputs, generated_images["fake_images_s"], self.fake_images_s)
                self.num_fake_inputs += 1

                # print(generated_images["fake_pool_t"].get_device())

                discriminator_results = self.model_discriminators(generated_images)
                # print(discriminator_results["prob_pred_mask_fake_t_is_real"].get_device())

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

                # Step optimizer
                self.generator_s_t_optimizer.step()

                all_losses["generator_s_t_loss"] = g1_generator_s_t_loss + g2_generator_s_t_loss


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

                all_losses["discriminator_t_loss"] = discriminator_t_loss_real + discriminator_t_loss_fake



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

                lsgan_loss_p = losses.generator_loss(discriminator_results["prob_pred_mask_t_is_real"])
                lsgan_loss_s_aux = losses.generator_loss(discriminator_results["prob_fake_s_aux_is_real"])

                g2_segmentation_t_loss = 0.1*g2_generator_t_s_loss + lsgan_loss_p_weight_value*lsgan_loss_p + 0.1*lsgan_loss_s_aux
                g2_segmentation_t_loss.backward(retain_graph=True)

                # Step optimizer
                self.segmentation_optimizer.step()

                all_losses["segmentation_t_loss"] = g1_segmentation_t_loss + g2_segmentation_t_loss


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

                all_losses["generator_t_s_loss"] = g1_generator_t_s_loss + g2_generator_t_s_loss
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

                all_losses["discriminator_s_loss"] = discriminator_s_loss_real + discriminator_s_aux_loss_real + pool_discriminator_s_loss

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

                all_losses["discriminator_p_loss"] = discriminator_p_loss_real + discriminator_p_loss_fake
                # End Train Loop

                if total_iter % self.print_freq == 0:
                    print(f'[train] epoch {epoch:02d}, iter {epoch_iter:03d}/{num_iter_per_epoch}')
                    for k in all_losses:
                        writer.add_scalar(f'train/loss/{k}', all_losses[k], total_iter)

                ##########################################################################################################################################################
            # Val Loop

            self.model_generators.eval()
            self.model_discriminators.eval()

            for idx, batch_sample in enumerate(self.dataloader_val):

                epoch_val_iter += 1
                total_val_iter += 1

                save_count += 1
                # print("Processing val batch {}".format(idx))

                val_images_s, val_gts_s = batch_sample['image_source'], batch_sample['gt_source']
                val_images_t, val_gts_t = batch_sample['image_target'], batch_sample['gt_target']

                # Executing each network with the current resources
                val_images_s = val_images_s.float().to(device)
                val_images_t = val_images_t.float().to(device)
                val_gts_s = val_gts_s.to(device)
                val_gts_t = val_gts_t.to(device)

                generated_val_images = self.model_generators(inputs = {"images_s": val_images_s, "images_t": val_images_t})

                # Adding all the synthesized images | fake images to a list for discriminator networks
                generated_val_images["fake_pool_t"] = self.fake_image_pool(self.num_fake_val_inputs, generated_val_images["fake_images_t"], self.fake_val_images_t)
                generated_val_images["fake_pool_s"] = self.fake_image_pool(self.num_fake_val_inputs, generated_val_images["fake_images_s"], self.fake_val_images_s)
                self.num_fake_val_inputs += 1

                discriminator_val_results = self.model_discriminators(generated_val_images)

                # ----------The Generator_S_T Network-----------

                # Get Loss specific to generator_s_t
                cycle_consistency_loss_s = \
                    self._lambda_s * losses.cycle_consistency_loss(
                        real_images=val_images_s.detach(),
                        generated_images=generated_val_images["cycle_images_s"].detach()
                )

                cycle_consistency_loss_t = \
                    self._lambda_t * losses.cycle_consistency_loss(
                        real_images=val_images_t.detach(),
                        generated_images=generated_val_images["cycle_images_t"].detach()
                )

                lsgan_loss_t = losses.generator_loss(discriminator_val_results["prob_fake_t_is_real"].detach())

                g1_generator_s_t_loss = cycle_consistency_loss_s + lsgan_loss_t
                g2_generator_s_t_loss = cycle_consistency_loss_t

                all_losses["generator_s_t_loss"] = g1_generator_s_t_loss + g2_generator_s_t_loss


                # ----------The Discriminator_T Network-----------

                discriminator_t_loss_real, discriminator_t_loss_fake = losses.discriminator_loss(prob_real_is_real=discriminator_val_results["prob_real_t_is_real"].detach(),
                                                                 prob_fake_is_real=discriminator_val_results["prob_fake_pool_t_is_real"].detach())

                all_losses["discriminator_t_loss"] = discriminator_t_loss_real + discriminator_t_loss_fake
                # ----------The Segmentation Network i.e E + C-----------

                # Get Loss specific to Segmentation

                ce_loss_t, dice_loss_t = losses.task_loss(generated_val_images["pred_mask_fake_t"].detach(), val_gts_s.detach())

                lsgan_loss_s = losses.generator_loss(discriminator_val_results["prob_fake_s_is_real"].detach())
                g1_generator_t_s_loss = cycle_consistency_loss_s
                g2_generator_t_s_loss = cycle_consistency_loss_t + lsgan_loss_s

                g1_segmentation_t_loss = ce_loss_t + dice_loss_t + 0.1*g1_generator_t_s_loss

                lsgan_loss_p = losses.generator_loss(discriminator_val_results["prob_pred_mask_t_is_real"].detach())
                lsgan_loss_s_aux = losses.generator_loss(discriminator_val_results["prob_fake_s_aux_is_real"]).detach()

                g2_segmentation_t_loss = 0.1*g2_generator_t_s_loss + lsgan_loss_p_weight_value*lsgan_loss_p + 0.1*lsgan_loss_s_aux

                all_losses["segmentation_t_loss"] = g1_segmentation_t_loss + g2_segmentation_t_loss

                # ----------The Generator_T_S Network i.e Decoder-----------

                # Get Loss specific to Decoder
                lsgan_loss_s = losses.generator_loss(discriminator_val_results["prob_fake_s_is_real"].detach())
                g1_generator_t_s_loss = cycle_consistency_loss_s
                g2_generator_t_s_loss = cycle_consistency_loss_t + lsgan_loss_s

                all_losses["generator_t_s_loss"] = g1_generator_t_s_loss + g2_generator_t_s_loss

                # ----------The Discriminator_S Network i.e Discriminator_AUX-----------

                # Get Loss specific to Decoder
                discriminator_s_loss_real, discriminator_s_loss_fake = losses.discriminator_loss(prob_real_is_real=discriminator_val_results["prob_real_s_is_real"].detach(),
                                                                 prob_fake_is_real=discriminator_val_results["prob_fake_pool_s_is_real"].detach())

                discriminator_s_aux_loss_real, discriminator_s_aux_loss_fake = losses.discriminator_loss(prob_real_is_real=discriminator_val_results["prob_cycle_s_aux_is_real"].detach(),
                                                                 prob_fake_is_real=discriminator_val_results["prob_fake_pool_s_aux_is_real"].detach())

                pool_discriminator_s_loss = discriminator_s_loss_fake + discriminator_s_aux_loss_fake

                all_losses["discriminator_s_loss"] = discriminator_s_loss_real + discriminator_s_aux_loss_real + pool_discriminator_s_loss

                # ----------The discriminator pred Network i.e Discriminator_MASK-----------

                # Get Loss specific to discriminator_mask
                discriminator_p_loss_real, discriminator_p_loss_fake = losses.discriminator_loss(prob_real_is_real=discriminator_val_results["prob_pred_mask_fake_t_is_real"].detach(),
                                                                 prob_fake_is_real=discriminator_val_results["prob_pred_mask_t_is_real"].detach())

                all_losses["discriminator_p_loss"] = discriminator_p_loss_real + discriminator_p_loss_fake
                # End Validate Loop

                if total_val_iter % self.print_freq == 0:
                    print(f'[val] epoch {epoch:02d}, iter {epoch_val_iter:03d}/{num_val_iter_per_epoch}')
                    for k in all_losses:
                        writer.add_scalar(f'val/loss/{k}', all_losses[k], total_val_iter)

                ##########################################################################################################################################################

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            self.adverserial_scheduler.step()

            writer.add_scalar(f'hyper_param/lr', curr_lr, epoch_iter)
            writer.add_scalar(f'hyper_param/seg_lr', curr_seg_lr, epoch_iter)

            # save network
            if epoch % self.save_epoch_freq == 0:
                torch.save(self.model_generators.cpu().state_dict(), os.path.join(self._checkpoint_dir, 'model_generators_{}.pth'.format(epoch)))
                torch.save(self.model_discriminators.cpu().state_dict(), os.path.join(self._checkpoint_dir, 'model_discriminators_{}.pth'.format(epoch)))

            self.model_generators = self.model_generators.to(device)
            self.model_discriminators = self.model_discriminators.to(device)

        writer.close()
        torch.save(self.model_generators.cpu().state_dict(), os.path.join(self._checkpoint_dir, 'latest_model_generators.pth'))
        torch.save(self.model_discriminators.cpu().state_dict(), os.path.join(self._checkpoint_dir, 'latest_model_discriminators.pth'))


def main(config_filename):

    with open(config_filename) as config_file:
        config = json.load(config_file)

    uda = UDA(config)
    uda.train()


if __name__ == '__main__':
    main(config_filename='./config.json')
