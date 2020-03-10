from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import random
import csv
from math import floor
from PIL import Image

from torch._six import int_classes as _int_classes
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
import torchvision.transforms.functional as TF

import nibabel as nib

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



class Two_idx_BatchSampler(Sampler):

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        source_iter, target_iter = self.sampler.iterate()

        self.admissible_length = min(len(source_iter), len(target_iter))
        for s_idx, t_idx in zip(source_iter, target_iter):
            batch.append((s_idx, t_idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.admissible_length) // self.batch_size
        else:
            return (self.admissible_length + self.batch_size - 1) // self.batch_size



class Two_idx_RandomSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        source_n, target_n = self.data_source._length_()
        source_iter_list = torch.randperm(source_n).tolist()
        target_iter_list = torch.randperm(target_n).tolist()
        return source_iter_list, target_iter_list

    def iterate(self):
        source_n, target_n = self.data_source._length_()
        source_iter_list = torch.randperm(source_n).tolist()
        target_iter_list = torch.randperm(target_n).tolist()
        return source_iter_list, target_iter_list


class Two_idx_SequentialSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        source_n, target_n = self.data_source._length_()
        source_iter = iter(range(source_n))
        target_iter = iter(range(target_n))
        return source_iter, target_iter



class CT_MR_Dataset(Dataset):
    """CT_MR Training Dataset"""

    def __init__(self,
                 source_path="mr_train_processed.csv",
                 target_path="ct_train_processed.csv",
                 augment_param=None):
        """
        Args:

        """

        self.source_path = source_path
        self.target_path = target_path

        self.source_list = pd.read_csv(self.source_path, header=None)
        self.source_images_dir = self.source_path.split('.')[0]
        self.target_list = pd.read_csv(self.target_path, header=None)
        self.target_images_dir = self.target_path.split('.')[0]

        self.transform = True if not augment_param is None else False
        self.augment_param = augment_param

        self.num_classes = 5


    def __len__(self):
        """
        len(dataset):

        """
        source_size = len(self.source_list)
        target_size = len(self.target_list)

        return min(source_size, target_size)
        # return [source_size, target_size]

    def _length_(self):
        """
        _length_(dataset):

        """
        source_size = len(self.source_list)
        target_size = len(self.target_list)

        return source_size, target_size

    def _decode_samples(self, image_path, label_path):

        image = torch.IntTensor(nib.load(image_path+".nii.gz").get_fdata())
        label = torch.IntTensor(nib.load(label_path+".nii.gz").get_fdata())

        return image, label

    def _load_samples(self, idxs):
        source_idx, target_idx = idxs

        source_img_path = os.path.join(self.source_images_dir, "slices", self.source_list.iloc[source_idx, 0])
        source_label_path = os.path.join(self.source_images_dir, "labels", self.source_list.iloc[source_idx, 1])

        target_img_path = os.path.join(self.target_images_dir, "slices", self.target_list.iloc[target_idx, 0])
        target_label_path = os.path.join(self.target_images_dir, "labels", self.target_list.iloc[target_idx, 1])

        source_img, source_label = self._decode_samples(source_img_path, source_label_path)
        target_img, target_label = self._decode_samples(target_img_path, target_label_path)

        return source_img, target_img, source_label, target_label

    def augmentations(self, image, label):
        image = TF.to_pil_image(image.squeeze())
        label = TF.to_pil_image(label.squeeze())

        angle = self.augment_param['rotation_angle']
        shift = self.augment_param['shift_range']
        shear = self.augment_param['shear_range']
        zoom = self.augment_param['zoom_range']

        # Random rotation
        if random.random() > 0.5:
            rotation = random.randint(-angle, angle)
            image = TF.rotate(image, rotation)
            label = TF.rotate(label, rotation)

        # Random Scale/Zoom
        if random.random() > 0.5:
            min_factor = 1
            max_factor = zoom
            factor = round(random.uniform(min_factor, max_factor), 2)

            def do(img):
                w, h = img.size

                img_zoomed = TF.resize(img, (int(round(img.size[0] * factor)),
                                             int(round(img.size[1] * factor))), interpolation=Image.NEAREST)
                w_zoomed, h_zoomed = img_zoomed.size

                return TF.center_crop(img_zoomed, 256)

            image, label = do(image), do(label)



        # Random affine transform
        if random.random() > 0.5:
            rotation = random.randint(-angle, angle) if angle > 0 else 0

            shift_x = round(random.uniform(-shift[0], shift[0]), 2) if shift[0] > 0 else 0
            shift_y = round(random.uniform(-shift[1], shift[1]), 2) if shift[1] > 0 else 0
            translate = (shift_x, shift_y)

            shear = round(random.uniform(-shear, shear), 2) if shear > 0 else 0
            scale = round(random.uniform(1, zoom), 2)

            image = TF.affine(image, rotation, translate, scale, shear)
            label = TF.affine(label, rotation, translate, scale, shear)

        image = TF.normalize(TF.to_tensor(image), mean=(0,), std=(1,))
        label = TF.to_tensor(label)

        return image, label

    def one_hot(self, gt):
        C = 5
        gt_extended=gt.clone().type(torch.long)

        intensities = [0, 205, 420, 500, 820] # intensities = gt_extended.unique().numpy()
        mapping = {c: t for c, t in zip(intensities, range(len(intensities)))}

        mask = torch.zeros(256, 256, dtype=torch.long).unsqueeze(0)
        for k in mapping:
            # Get all indices for current class
            idx = (gt_extended==torch.tensor(k, dtype=torch.long))
            mask[idx] = torch.tensor(mapping[k], dtype=torch.long)

        # print("Mask Shape", mask.shape)
        # print("Mask Unique", mask.unique())

        one_hot = torch.FloatTensor(C, gt_extended.size(1), gt_extended.size(2)).zero_()
        one_hot.scatter_(0, mask, 1)
        return one_hot


    def __getitem__(self, idxs):

        image_source, image_target, gt_source, gt_target = self._load_samples(idxs)

        if self.transform:
            image_source, gt_source = self.augmentations(image_source, gt_source)
            image_target, gt_target = self.augmentations(image_target, gt_target)

        gt_source = self.one_hot(gt_source)
        gt_target = self.one_hot(gt_target)

        sample = {'image_source': image_source, 'gt_source': gt_source, 'image_target': image_target, 'gt_target': gt_target}

        return sample


if __name__ == "__main__":

    source_path = "data/ct_mr_train/pre_processed/mr_train_processed.csv"
    target_path="data/ct_mr_train/pre_processed/ct_train_processed.csv"

    rotation_angle = 15
    shift_range = [0.3,0.3]
    shear_range = 0.1
    zoom_range = 1.3

    basic_augmentations = {'rotation_angle': rotation_angle,
                           'shift_range': shift_range,
                           'shear_range': shear_range,
                           'zoom_range': zoom_range }

    dataset = CT_MR_Dataset(source_path, target_path, augment_param=basic_augmentations)

    sizes = len(dataset)

    print("Length of dataset: ", dataset._length_())

    two_idx_sampler = Two_idx_RandomSampler(dataset)
    two_idx_batch_sampler = Two_idx_BatchSampler(two_idx_sampler, batch_size=8, drop_last=False)

    dataloader = DataLoader(dataset, batch_sampler=two_idx_batch_sampler, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch,"/",len(two_idx_batch_sampler))

        # print("Source_image: ", sample_batched['image_source'].shape)
        # print("source_label: ", sample_batched['gt_source'].shape)
        # print("Target_image: ", sample_batched['image_target'].shape)
