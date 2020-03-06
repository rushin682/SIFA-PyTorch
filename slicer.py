from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF

import nibabel as nib

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class CT_MR_Train(Dataset):
    """CT_MR Training Dataset"""

    def __init__(self, root_dir="data/ct_mr_train",
                 modality="MR",
                 images_path="mr_train",
                 images_list="mr_train.csv",
                 transform=None,
                 augment_param=None):
        """
        Args:

        """

        self.root_dir = root_dir

        self.modality = modality

        self.images_path = images_path
        self.images_list = images_list
        self.images_frame = pd.read_csv(os.path.join(self.root_dir, images_list), header=None)

        self.transform = True
        self.variable_size = 256

        self.normalize = transforms.Normalize((0,), (1,), inplace=False)

        self.augment_param = augment_param
        self.count = 0

        self.preprocessed_dir = os.path.join(self.root_dir, "pre_processed")
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)


    def __len__(self):
        """
        len(dataset):

        """
        dataset_size = len(self.images_frame) # MR-Scans

        return dataset_size

    def save_file(self, image, label, file_name):
        # save to the disk
        self.count += 1

        image_name = "coronal_slice_"+"{:04d}".format(self.count)
        label_name = "coronal_slice_label"+"{:04d}".format(self.count)

        export_dir = os.path.join(self.preprocessed_dir, self.images_list.split('.')[0]+"_processed")
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            os.makedirs(os.path.join(export_dir, "slices"))
            os.makedirs(os.path.join(export_dir, "labels"))

        # image_dir = os.path.join(export_dir, file_name)
        # if not os.path.exists(image_dir):
        #     os.makedirs(image_dir)
        #     os.makedirs(os.path.join(image_dir, "slices"))
        #     os.makedirs(os.path.join(image_dir, "labels"))

        file_name = os.path.join(self.preprocessed_dir, self.images_list.split('.')[0]+"_processed.csv")

        with open(file_name, 'a', newline='') as csvfile:
            wrt = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            wrt.writerow([image_name, label_name])

        nii_image = nib.Nifti1Image(image.numpy(), np.eye(4))
        nii_label = nib.Nifti1Image(label.numpy(), np.eye(4))


        nib.save(nii_image, os.path.join(export_dir, "slices", image_name+".nii.gz"))
        nib.save(nii_label, os.path.join(export_dir, "labels", label_name+".nii.gz"))




    def get_shape(self, image, axis):

        if axis==0:
            self.variable_size = int((image.shape[1] + image.shape[2])/2)

        elif axis==1:
            self.variable_size = int((image.shape[0] + image.shape[2])/2)

        elif axis==2:
            self.variable_size = int((image.shape[0] + image.shape[1])/2)

        return (image.shape[axis],256,256)


    def perform_augmentations(self, image, label):

        image = TF.to_pil_image(image, mode='I')
        label = TF.to_pil_image(label, mode='I')

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
            scale = random.uniform(1, zoom) if zoom > 1 else 1
            size = -128*(scale) + 384
            image = TF.center_crop(image, size)
            image = TF.resize(image, 256)

            label = TF.center_crop(label, size)
            label = TF.resize(label, 256)


        # Random affine transform
        if random.random() > 0.5:
            rotation = random.randint(-angle, angle) if angle > 0 else 0

            shift_x = random.uniform(-shift[0], shift[0]) if shift[0] > 0 else 0
            shift_y = random.uniform(-shift[1], shift[1]) if shift[1] > 0 else 0
            translate = (shift_x, shift_y)

            shear = random.uniform(-shear, shear) if shear > 0 else 0
            scale = zoom

            image = TF.affine(image, rotation, translate, scale, shear)
            label = TF.affine(label, rotation, translate, scale, shear)


        return TF.to_tensor(image), TF.to_tensor(label)


    def perform_transformations(self, image, label, axis, image_name):
        shape = self.get_shape(image, axis)

        if axis==0:
            first = 80
            last = 176
        elif axis==1:
            first = 146
            last = 408
        elif axis==2:
            first = 24
            last = 87

        tr_image = torch.tensor(()).new_empty(shape)
        tr_label = torch.tensor(()).new_empty(shape)

        self.transform = transforms.Compose([
                    transforms.ToPILImage(mode='I'),
                    transforms.CenterCrop(size=self.variable_size),
                    transforms.Resize(size=256),
                    transforms.ToTensor()
                    ])

        last = min(last, image.shape[axis])
        for j in range(first, last):

            if axis==0:
                tr_slice = self.normalize(self.transform(image[j,:,:]))
                tr_label_slice = self.transform(label[j,:,:])

            elif axis==1:
                tr_slice = self.normalize(self.transform(image[:,j,:]))
                tr_label_slice = self.transform(label[:,j,:])
            elif axis==2:
                tr_slice = self.normalize(self.transform(image[:,:,j]))
                tr_label_slice = self.transform(label[:,:,j])

            # if not self.augment_param is None:
            #     tr_slice, tr_label_slice = self.perform_augmentations(tr_slice, tr_label_slice)

            tr_image[j,:,:] = tr_slice
            tr_label[j,:,:] = tr_label_slice

            self.save_file(tr_slice, tr_label_slice, image_name)

        return tr_image, tr_label



    def __getitem__(self, idx):

        image_path = os.path.join(self.root_dir,
                                  self.images_path,
                                  self.images_frame.iloc[idx, 0])

        label_path = os.path.join(self.root_dir,
                                  self.images_path,
                                  self.images_frame.iloc[idx, 1])


        image_name = self.images_frame.iloc[idx, 0]
        image = np.int32(nib.load(image_path).get_fdata())
        label = np.int32(nib.load(label_path).get_fdata())
        axis = self.images_frame.iloc[idx, 2]

        print(image.shape)

        if self.transform:
            image, label = self.perform_transformations(image, label, axis, image_name)
            print(self.variable_size)

        sample = {'image_name': self.images_frame.iloc[idx, 0], 'label_name': self.images_frame.iloc[idx, 1], 'image': image, 'label': label, 'axis': axis}

        return sample


if __name__ == "__main__":

    root_dir="data/ct_mr_train"
    modality = "CT"
    if modality=="MR":
        images_path="mr_train"
        images_list="mr_train.csv"

    elif modality=="CT":
        images_path="ct_train"
        images_list="ct_val.csv"

    basic_transforms = transforms.Compose([
                transforms.ToPILImage(mode='I'),
                transforms.CenterCrop(size=256),
                transforms.ToTensor()
                ])

    rotation_angle = 15
    shift_range = [0.3,0.3]
    shear_range = 0.1
    zoom_range = 1.3

    basic_augmentations = {'rotation_angle': rotation_angle, 'shift_range': shift_range, 'shear_range': shear_range, 'zoom_range': zoom_range }

    dataset = CT_MR_Train(root_dir,modality,images_path,images_list,transform=False, augment_param=None)

    print("Length of dataset: ", len(dataset))

    for i in range(len(dataset)):
        sample = dataset[i]
        print(i)
