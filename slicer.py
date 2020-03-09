from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import random
import csv
from PIL import Image

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

    def save_file(self, image, label):
        # save to the disk
        self.count += 1

        image_name = "coronal_slice_"+"{:04d}".format(self.count)
        label_name = "coronal_slice_label"+"{:04d}".format(self.count)

        export_dir = os.path.join(self.preprocessed_dir, self.images_list.split('.')[0]+"_processed")
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            os.makedirs(os.path.join(export_dir, "slices"))
            os.makedirs(os.path.join(export_dir, "labels"))

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


    def perform_transformations(self, image, label, axis):
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

        tr_image = torch.IntTensor(()).new_empty(shape)
        tr_label = torch.IntTensor(()).new_empty(shape)

        self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.CenterCrop(size=self.variable_size),
                    transforms.Resize(size=256, interpolation=Image.NEAREST),
                    transforms.ToTensor()
                    ])

        self.normalize = transforms.Normalize((0,), (1,), inplace=False)

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


            tr_image[j,:,:] = tr_slice
            tr_label[j,:,:] = tr_label_slice

            self.save_file(tr_slice, tr_label_slice)

        return tr_image, tr_label



    def __getitem__(self, idx):

        image_path = os.path.join(self.root_dir,
                                  self.images_path,
                                  self.images_frame.iloc[idx, 0])

        label_path = os.path.join(self.root_dir,
                                  self.images_path,
                                  self.images_frame.iloc[idx, 1])


        image = torch.IntTensor(nib.load(image_path).get_fdata())
        label = torch.IntTensor(nib.load(label_path).get_fdata())
        axis = self.images_frame.iloc[idx, 2]

        # print(label.unique()) # Got my lead. I have to back track from here. Thank You

        print(image.shape)

        if self.transform:
            image, label = self.perform_transformations(image, label, axis)

        sample = {'image': image, 'label': label, 'axis': axis}

        return sample


if __name__ == "__main__":

    root_dir="data/ct_mr_train"
    modality = "CT"
    if modality=="MR":
        images_path="mr_train"
        images_list="mr_val.csv"

    elif modality=="CT":
        images_path="ct_train"
        images_list="ct_val.csv"

    dataset = CT_MR_Train(root_dir,modality,images_path,images_list,transform=False, augment_param=None)

    print("Length of dataset: ", len(dataset))


    for i in range(len(dataset)):
        sample = dataset[i]
        print(i)
