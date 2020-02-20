import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torchsummary import summary

from layers import Convolution2D, DilatedConv2D, Deconvolution2D
from modules import Residual_block
import json


class Encoder(nn.Module):
    def __init__(self, skip_conn=False, is_training=True, dropout_rate=0.75):
        super(Encoder, self).__init__()

        ech = 16 # Encoder minimum channel multiple
        self.padding = "constant"
        self.dropout_rate = dropout_rate

        self.encoder = nn.Sequential(OrderedDict([
                          ("C16", Convolution2D(input_ch, ech,
                                                kernel_size=7, stride=1,
                                                padding_mode="same",
                                                norm_type="Batch",
                                                dropout_rate=self.dropout_rate)),
                          ("R16", Residual_block(ech, ech,
                                                 padding=self.padding,
                                                 norm_type="Batch",
                                                 dropout_rate=self.dropout_rate)),
                          ("M1", nn.MaxPool2d(kernel_size=2, stride=2,
                                             padding=1)),


                          ("R32", Residual_block(ech, ech*2,
                                                 padding=self.padding,
                                                 norm_type="Batch",
                                                 dropout_rate=self.dropout_rate)),
                          ("M2", nn.MaxPool2d(kernel_size=2, stride=2,
                                             padding=1)),


                          ("R64_1", nn.Residual_block(ech*2, ech*4,
                                                      padding=self.padding,
                                                      norm_type="Batch",
                                                      dropout_rate=self.dropout_rate)),
        ]))
