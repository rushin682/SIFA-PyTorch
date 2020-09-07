import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data.tf_loader import _load_sample
from PIL import Image
import random


class MMWHSDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    It requires two directories to host training images from domain S '/path/to/data/trainS'
    and from domain T '/path/to/data/trainT' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testS' and '/path/to/data/testT' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_S = os.path.join(opt.dataroot, opt.phase + 'S')  # create a path '/path/to/data/trainS'
        self.dir_T = os.path.join(opt.dataroot, opt.phase + 'T')  # create a path '/path/to/data/trainT'

        self.S_paths = sorted(make_dataset(self.dir_S, opt.max_dataset_size))   # load image paths from '/path/to/data/trainS'
        self.T_paths = sorted(make_dataset(self.dir_T, opt.max_dataset_size))    # load image paths from '/path/to/data/trainT'
        self.S_size = len(self.S_paths)  # get the size of dataset S
        self.T_size = len(self.T_paths)  # get the size of dataset T
        ttoS = self.opt.direction == 'TtoS'
        input_nc = self.opt.output_nc if ttoS else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if ttoS else self.opt.output_nc      # get the number of channels of output image
        self.transform_S = get_transform(self.opt, grayscale=(input_nc == 1)) # maybe method=Image.BILINEAR instead of BICUBIC
        self.transform_T = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains S, T, S_paths and T_paths
            S (tensor)       -- an image in the input domain
            T (tensor)       -- its corresponding image in the target domain
            S_paths (str)    -- image paths
            T_paths (str)    -- image paths
        """
        S_path = self.S_paths[index % self.S_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_T = index % self.T_size
        else:   # randomize the index for domain T to avoid fixed pairs.
            index_T = random.randint(0, self.T_size - 1)
        T_path = self.T_paths[index_T]
        # S_img = Image.open(S_path).convert('RGB') # Instead call tfloader.py (Get tfrecord decoded)
        # T_img = Image.open(T_path).convert('RGB') # Instead call tfloader.py (Get tfrecord decoded)

        S_img, S_label = _load_sample(S_path)
        T_img, T_label = _load_sample(T_path)

        # apply image transformation
        S_img = self.transform_S(S_img)
        S_label = self.transform_S(S_label)

        T_img = self.transform_T(T_img)
        T_label = self.transform_T(T_label)

        return {'S': S_img, 'gt_S' : S_label, 'T': T_img, 'gt_T' : T_label, 'S_paths': S_path, 'T_paths': T_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.S_size, self.T_size)

    def class_count(self): # Not Sure if this will work, but I hope it does.
        return 5
