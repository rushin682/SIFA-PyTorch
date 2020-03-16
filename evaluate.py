import time
from datetime import datetime
import json
import numpy as np
import random
import os
import csv
import pandas as pd
from math import floor, ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as tvu
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import nibabel as nib
import medpy.metric.binary as mmb

import losses
from model import SIFA_generators, SIFA_discriminators

from stats_func import *

class CT_MR_TestSet(Dataset):
    """CT_MR Testing Dataset"""
    def __init__(self,
                 data_path="data/MMWHS/ct_mr_dataset",
                 test_path="ct_test.csv",
                 transforms=False):
        """
        Args:

        """

        self.data_path = data_path
        self.test_path = os.path.join(self.data_path, "datalist", test_path)

        self.test_list = pd.read_csv(self.test_path, header=None)
        self.test_images_dir = os.path.join(self.data_path, test_path.split('.')[0]) # folder path of test files

        self.modality = "ct" # Could be mr

        self.transforms = transforms

        self.num_classes = 5

    def __len__(self):
        """
        len(dataset):

        """
        return len(self.test_list)

    def _decode_samples(self, image_path, label_path):

        image = np.transpose(nib.load(image_path+".nii.gz").get_fdata(), axes=(2,0,1))
        label = np.transpose(nib.load(label_path+".nii.gz").get_fdata(), axes=(2,0,1))

        if self.modality=="ct":
            # {-2.8, 3.2} need to be changed according to the data statistics
            image = np.subtract(np.multiply(np.divide(np.subtract(image, -2.8), np.subtract(3.2, -2.8)), 2.0),1)

        elif self.modality=="mr":
            # {-1.8, 4.4} need to be changed according to the data statistics
            iamge = np.subtract(np.multiply(np.divide(np.subtract(image, -1.8), np.subtract(4.4, -1.8)), 2.0),1)

        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)

        return image, label

    def _load_samples(self, idx):

        test_img_path = os.path.join(self.test_images_dir, self.test_list.iloc[idx, 0])
        test_label_path = os.path.join(self.test_images_dir, self.test_list.iloc[idx, 1])

        test_img, test_label = self._decode_samples(test_img_path, test_label_path)

        return test_img, test_label

    def perform_transforms(self, image, gt):

        if True:
            image = torch.flip(image, dims=[1,2])
            gt = torch.flip(gt, dims=[1,2])

        return image, gt

    def __getitem__(self, idx):

        image, gt = self._load_samples(idx)

        if self.transforms:
            image, gt = self.perform_transforms(image, gt)

        dummy = torch.rand((1,256,256))

        sample = {'image': image, 'gt': gt, 'dummy': dummy}

        return sample




class UDA_EVAL:
    "The Evaluation Module"

    def __init__(self, config):

        self._data_path = config['data_path']
        self._output_root_dir = config['output_root_dir']
        if not os.path.isdir(self._output_root_dir):
            os.makedirs(self._output_root_dir)

        self._pool_size = int(config['pool_size'])

        self._skip_conn = bool(config['skip_conn'])
        self._num_cls = int(config['num_cls'])


        self._dropout_rate = float(config['dropout_rate'])
        self._is_training = bool(config['is_training'])
        self._batch_size = int(config['batch_size'])
        self._checkpoint_dir = config['checkpoint_dir']
        self.print_freq = config['print_frequency']
        self.save_epoch_freq = config['save_epoch_freq']

        self.ngpu = config['ngpu']

        self._test_path = config['test_path']
        self.transforms = config['transforms']
        self._test_dir = os.path.join(self._data_path, self._test_path.split('.')[0])

        self.data_size = [1, 256, 256]
        self.label_size = [1, 256, 256]

        self.contour_map = {
                            "bg": 0,
                            "la_myo": 1,
                            "la_blood": 2,
                            "lv_blood": 3,
                            "aa": 4
                            }

    def load_param_dict(self, path):
        dict = torch.load(path)
        altered_dict = {'.'.join(key.split('.')[1:]):value for key,value in dict.items()}
        return altered_dict

    def one_hot(self, gt):
        C = 5
        gt_extended=gt.clone().type(torch.long)
        gt_extended = gt_extended.unsqueeze(1)
        # intensities = [0, 1, 2, 3, 4] # intensities = gt_extended.unique().numpy()
        # mapping = {c: t for c, t in zip(intensities, range(len(intensities)))}

        # mask = torch.zeros(256, 256, dtype=torch.long).unsqueeze(0)
        # for k in mapping:
            # Get all indices for current class
            # idx = (gt_extended==torch.tensor(k, dtype=torch.long))
            # mask[idx] = torch.tensor(mapping[k], dtype=torch.long)

        # print("Mask Shape", mask.shape)
        # print("Mask Unique", mask.unique())

        one_hot = torch.FloatTensor(gt_extended.size(0), C, gt_extended.size(2), gt_extended.size(3)).zero_()
        one_hot.scatter_(1, gt_extended, 1)
        return one_hot


    def test(self):
        "Test Function"

        # -------------------- SETTINGS: DATASET BUILDERS
        # Load Dataset from dataloader
        test_dataset = CT_MR_TestSet(self._data_path, self._test_path, transforms=self.transforms)
        self.dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

        # -------------------- SETTINGS: NETWORK ARCHITECTURE
        device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

        self.model_generators = SIFA_generators(skip_conn=self._skip_conn, is_training=self._is_training, dropout_rate=self._dropout_rate)
        self.model_generators.load_state_dict(self.load_param_dict(os.path.join(self._checkpoint_dir, 'latest_model_generators.pth')))

        self.model_discriminators = SIFA_discriminators()
        self.model_discriminators.load_state_dict(self.load_param_dict(os.path.join(self._checkpoint_dir, 'latest_model_discriminators.pth')))

        self.model_generators = self.model_generators.to(device)
        self.model_discriminators = self.model_discriminators.to(device)


        # -------------------- Dice & ASSD score initializers
        dice_list = []
        assd_list = []

        # Test Loop
        self.model_generators.eval()
        self.model_discriminators.eval()
        for idx, sample in enumerate(test_dataset):
            print("Sample #", idx)
            image, label, dummy = sample['image'], sample['gt'], sample['dummy']

            image = image.to(device).unsqueeze(0)
            dummy = dummy.to(device).unsqueeze(0)
            label = label.to(device).unsqueeze(0)

            print("Shapes: ",image.shape)

            tmp_pred = torch.zeros(size=label.shape)

            for ii in range(int(np.floor(image.shape[1]))):

                print("Processing Slice:", ii)
                generated_images = self.model_generators(inputs = {"images_s": dummy[:,0,:,:].unsqueeze(1), "images_t": image[:,ii,:,:].unsqueeze(1)})

                generated_images["fake_pool_t"] = generated_images["fake_images_t"]
                generated_images["fake_pool_s"] = generated_images["fake_images_s"]

                discriminator_results = self.model_discriminators(generated_images)

                pred_mask_t = generated_images['pred_mask_t']
                predictor_t = nn.Softmax2d()(pred_mask_t)
                compact_pred_t, indices = torch.max(predictor_t, dim=1)

                label = self.one_hot(label[:,ii,:,:])

                tmp_pred[:,ii,:,:] = compact_pred_t.clone()

            for c in range(1, self._num_cls):
                pred_test_data_tr = tmp_pred.clone().numpy()
                pred_test_data_tr[pred_test_data_tr != c] = 0

                pred_gt_data_tr = label.clone().numpy()
                pred_gt_data_tr[pred_gt_data_tr != c] = 0

                dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))
                assd_list.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))

        # Dice Scores
        dice_arr = 100 * np.reshape(dice_list, [4,-1]).transpose()

        dice_mean = np.mean(dice_arr, axis=1)
        dice_std = np.std(dice_arr, axis=1)

        print ("Dice:")
        print (f"AA: {dice_mean[3]:02d}({dice_std[3]:02d})")
        print (f"LAC: {dice_mean[1]:02d}({dice_std[1]:02d})")
        print (f"LVC: {dice_mean[2]:02d}({dice_std[2]:02d})")
        print (f"Myo: {dice_mean[0]:02d}({dice_std[0]:02d})")
        print (f"Mean: {np.mean(dice_mean):02d}")

        # ASSD Scores
        assd_arr = np.reshape(add_list, [4, -1]).transpose()

        assd_mean = np.mean(assd_arr, axis=1)
        assd_std = np.std(assd_arr, axis=1)

        print ("ASSD:")
        print (f"AA: {assd_mean[3]:02d}({assd_std[3]:02d})")
        print (f"LAC: {assd_mean[1]:02d}({assd_std[1]:02d})")
        print (f"LVC: {assd_mean[2]:02d}({assd_std[2]:02d})")
        print (f"Myo: {assd_mean[0]:02d}({assd_std[0]:02d})")
        print (f"Mean: {np.mean(assd_mean):02d}")


















def main(config_filename):

    with open(config_filename) as config_file:
        config = json.load(config_file)

    uda = UDA_EVAL(config)
    uda.test()

if __name__ == '__main__':
    main(config_filename="./evaluation_config.json")
