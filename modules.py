import torch
import torch.nn.functional as F
import torch.nn as nn

from layers import Convolution2D, Deconvolution2D, DilatedConv2D

class Residual_block(nn.module):
    def __init__(self, input_ch, output_ch, padding='REFLECT'):
        super(Residual_block, self).__init__()

        self.conv1 = Convolution2D(input_ch, output_ch, kernel_size=3, stride=1, norm_type='Ins')
        self.conv2 = Convolution2D(output_ch, output_ch, kernel_size=3, stride=1, do_relu=False, norm_type='Ins')

        self.skip = torch.nn.ReLU()

    def forward(self, input):
        #Some padding
        output = self.conv1(padded_inp)

        #Some padding
        output = self.conv2(output)
        output = self.skip(output+input)
