import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torchsummary import summary

from layers import Convolution2D, DilatedConv2D, Deconvolution2D
from modules import Residual_block, Dilated_Residual_block
import json


class Encoder(nn.Module):
    def __init__(self, input_ch, skip_conn=False, dropout_rate=0.75, is_training=True):
        super(Encoder, self).__init__()

        ech = 16 # Encoder minimum channel multiple
        self.padding = "constant"

        dropout_rate = dropout_rate
        is_training = is_training

        self.residual_chunk = nn.Sequential(OrderedDict([

                                ("C16", Convolution2D(input_ch, ech,
                                                      kernel_size=7, stride=1,
                                                      padding_mode="same",
                                                      norm_type="Batch",
                                                      dropout_rate=dropout_rate,
                                                      is_training=is_training)),
                                ("R16", Residual_block(ech, ech,
                                                       padding=self.padding,
                                                       norm_type="Batch",
                                                       dropout_rate=dropout_rate,
                                                       is_training=is_training)),
                                ("M1", nn.MaxPool2d(kernel_size=2, stride=2,
                                                    padding=1)),

                                ("R32", Residual_block(ech, ech*2,
                                                       padding=self.padding,
                                                       norm_type="Batch",
                                                       dropout_rate=dropout_rate,
                                                       is_training=is_training,
                                                       channel_pad=True)),
                                ("M2", nn.MaxPool2d(kernel_size=2, stride=2, padding=1)),


                                ("R64_1", Residual_block(ech*2, ech*4,
                                                         padding=self.padding,
                                                         norm_type="Batch",
                                                         dropout_rate=dropout_rate,
                                                         is_training=is_training,
                                                         channel_pad=True)),
                                ("R64_2", Residual_block(ech*4, ech*4,
                                                         padding=self.padding,
                                                         norm_type="Batch",
                                                         dropout_rate=dropout_rate,
                                                         is_training=is_training)),
                                ("M3", nn.MaxPool2d(kernel_size=2, stride=2, padding=1)),


                                ("R128_1", Residual_block(ech*4, ech*8,
                                                          padding=self.padding,
                                                          norm_type="Batch",
                                                          dropout_rate=dropout_rate,
                                                          is_training=is_training,
                                                          channel_pad=True)),
                                ("R128_2", Residual_block(ech*8, ech*8,
                                                          padding=self.padding,
                                                          norm_type="Batch",
                                                          dropout_rate=dropout_rate,
                                                          is_training=is_training)),

                                ("R256_1", Residual_block(ech*8, ech*16,
                                                          padding=self.padding,
                                                          norm_type="Batch",
                                                          dropout_rate=dropout_rate,
                                                          is_training=is_training,
                                                          channel_pad=True)),
                                ("R256_2", Residual_block(ech*16, ech*16,
                                                          padding=self.padding,
                                                          norm_type="Batch",
                                                          dropout_rate=dropout_rate,
                                                          is_training=is_training)),
                                ("R256_3", Residual_block(ech*16, ech*16,
                                                          padding=self.padding,
                                                          norm_type="Batch",
                                                          dropout_rate=dropout_rate,
                                                          is_training=is_training)),
                                ("R256_4", Residual_block(ech*16, ech*16,
                                                          padding=self.padding,
                                                          norm_type="Batch",
                                                          dropout_rate=dropout_rate,
                                                          is_training=is_training)),

                                ("R512_1", Residual_block(ech*16, ech*32,
                                                          padding=self.padding,
                                                          norm_type="Batch",
                                                          dropout_rate=dropout_rate,
                                                          is_training=is_training,
                                                          channel_pad=True)),
                                ("R512_2", Residual_block(ech*32, ech*32,
                                                          padding=self.padding,
                                                          norm_type="Batch",
                                                          dropout_rate=dropout_rate,
                                                          is_training=is_training))
        ]))

        self.dilation_chunk = nn.Sequential(OrderedDict([

                                  ("D512_1", Dilated_Residual_block(ech*32, ech*32,
                                                                    padding=self.padding,
                                                                    norm_type="Batch",
                                                                    dropout_rate=dropout_rate,
                                                                    is_training=is_training)),
                                  ("D512_2", Dilated_Residual_block(ech*32, ech*32,
                                                                    padding=self.padding,
                                                                    norm_type="Batch",
                                                                    dropout_rate=dropout_rate,
                                                                    is_training=is_training)),

                                  ("C512_1", Convolution2D(ech*32, ech*32,
                                                           kernel_size=3, stride=1,
                                                           padding_mode="same",
                                                           norm_type="Batch",
                                                           dropout_rate=dropout_rate,
                                                           is_training=is_training)),
                                  ("C512_2", Convolution2D(ech*32, ech*32,
                                                           kernel_size=3, stride=1,
                                                           padding_mode="same",
                                                           norm_type="Batch",
                                                           dropout_rate=dropout_rate,
                                                           is_training=is_training))
        ]))

    def forward(self, input):

        residual_output = self.residual_chunk(input)

        dilation_output = self.dilation_chunk(residual_output)

        return dilation_output, residual_output

if __name__ == "__main__":
    model = Encoder(input_ch = 1, skip_conn = True)
    summary(model, input_size=(1, 256, 256))
