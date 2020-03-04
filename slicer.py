from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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
                 transform=None):
        """
        Args:
            root_dir:
            transform:

            Maybe:
                num_of_files:
                num_of_slices:
        """

        self.root_dir = root_dir

        self.modality = modality

        self.images_path = images_path
        self.images_frame = pd.read_csv(os.path.join(self.root_dir, images_list), header=None)

        self.transform = transform
        self.normalize = transforms.Normalize(mean=0, std=1, inplace=False)

    def __len__(self):
        """
        len(dataset):
            returns (the size of the dataset(Source and Target),
                     the coronal_axis of Source(MR),
                     the coronal_axis of Target(CT)
        """
        dataset_size = len(self.images_frame) # MR-Scans

        return dataset_size

    def get_shape(self, image, axis):

        return (image.shape[axis],256,256)

    def perform_transformations(self, image, label, axis):
        shape = self.get_shape(image, axis)

        tr_image = torch.tensor(()).new_empty(shape)
        tr_label = torch.tensor(()).new_empty(shape)

        for j in range(image.shape[axis]):

            if axis==0:
                tr_image[j,:,:] = self.transform(image[j,:,:])
                tr_label[j,:,:] = self.transform(label[j,:,:])
            elif axis==1:
                tr_image[j,:,:] = self.transform(image[:,j,:])
                tr_image[j,:,:] = self.transform(label[:,j,:])
            elif axis==2:
                tr_image[j,:,:] = self.transform(image[:,:,j])
                tr_image[j,:,:] = self.transform(label[:,:,j])

        return tr_image, tr_label

    def __getitem__(self, idx):

        image_path = os.path.join(self.root_dir,
                                  self.images_path,
                                  self.images_frame.iloc[idx, 0])

        label_path = os.path.join(self.root_dir,
                                  self.images_path,
                                  self.images_frame.iloc[idx, 1])

        image = np.int32(nib.load(image_path).get_fdata())
        label = np.int32(nib.load(label_path).get_fdata())
        axis = self.images_frame.iloc[idx, 2]


        if self.transform:
            image, label = self.perform_transformations(image, label, axis)

        sample = {'image_name': self.images_frame.iloc[idx, 0], 'label_name': self.images_frame.iloc[idx, 1], 'image': image, 'label': label, 'axis': axis}

        return sample


if __name__ == "__main__":

    root_dir="data/ct_mr_train"
    modality = "MR"
    if modality=="MR":
        images_path="mr_train"
        images_list="mr_train.csv"

    elif modality=="CT":
        images_path="ct_train"
        images_list="ct_train.csv"

    transformations = transforms.Compose([
                transforms.ToPILImage(mode='I'),
                transforms.CenterCrop(size=256),
                transforms.ToTensor()])

    dataset = CT_MR_Train(root_dir,modality,images_path,images_list,transform=transformations)

    print("Length of dataset: ", len(dataset))

    for i in range(len(dataset)):
        sample = dataset[i]
        print(i)
