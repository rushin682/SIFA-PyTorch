import os
import sys
import time
from . import util

import numpy as np

import torch
import torchvision
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

def save_images(visuals, image_path, width):
    """Save images to the disk.
    Parameters:
        visuals (OrderedDict) -- an ordered dictionary that stores (name, images) pairs
                                 [Image to be saved. If given a mini-batch tensor, saves the tensor as a grid of images by calling make_grid]
        image_path (str)      -- the string is used to create image paths
        width (maybe)         -- the images will be resized to width x width
    This function will save images stored in 'visuals' to the directory specified in the image path
    """

    for label, im_data in visuals.items():
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        save_image(im_data, save_path)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.
    It uses a Python library 'tensorboard' for creating images and displaying them to an HTML file.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class
        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: Initialize tensorboard writer (create server object)
        """

        self.opt = opt # cache the option
        self.name = opt.name
        self.saved = False

        self.writer = SummaryWriter(log_dir = opt.name)


    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on tensorboard display;
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results
        """

        images = []
        idx = 0
        for label, image in visuals.items():
            self.writer.add_images(tag=label, img_tensor=image, global_step=epoch)
            if save_result:
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                save_image(image, img_path)

    def plot_current_losses(self, epoch, losses):
        """display the current losses on tensorboard display: dictionary of error labels and values
        Parameters:
            epoch (int)           -- current epoch
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        print(f'[train] epoch {epoch:02:d}') # , iter {epoch_iter:03d}/{num_iter_per_epoch}')
        for k in losses:
            self.writer.add_scalar(f'train/loss/{k}', losses[k], epoch)
