import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torchsummary import summary

from layers import Convolution2D, DilatedConv2D, Deconvolution2D
from modules import Residual_block
import json

'''
with open('./config_param.json') as config_file:
    config = json.load(config_file)

BATCH_SIZE = int(config['batch_size'])
POOL_SIZE = int(config['pool_size'])

# The height of each image.
IMG_HEIGHT = 256

# The width of each image.
IMG_WIDTH = 256
'''

# class SIFA(nn.Module):
#     def __init__(self,)
#
#         self.generator_s_t = Generator_S_T()
#
#     def forward(self, input):
#
#         # return ...



class Generator_S_T(nn.Module):

    def __init__(self, input_ch, skip_conn=False):
        super(Generator_S_T, self).__init__()

        gch = 32 # generator minimum channel multiple
        self.padding = "constant"
        self.skip_conn = skip_conn

        self.generator = nn.Sequential(OrderedDict([
                            ("conv1", Convolution2D(input_ch, gch,
                                                    kernel_size=7, stride=1,
                                                    norm_type="Ins")),
                            ("conv2", Convolution2D(gch, gch*2,
                                                    kernel_size=3, stride=2,
                                                    padding_mode="same",
                                                    norm_type="Ins")),
                            ("conv3", Convolution2D(gch*2, gch*4,
                                                    kernel_size=3, stride=2,
                                                    padding_mode="same",
                                                    norm_type="Ins")),

                            ("res_block1", Residual_block(gch*4, gch*4, padding=self.padding)),
                            ("res_block2", Residual_block(gch*4, gch*4, padding=self.padding)),
                            ("res_block3", Residual_block(gch*4, gch*4, padding=self.padding)),
                            ("res_block4", Residual_block(gch*4, gch*4, padding=self.padding)),
                            ("res_block5", Residual_block(gch*4, gch*4, padding=self.padding)),
                            ("res_block6", Residual_block(gch*4, gch*4, padding=self.padding)),
                            ("res_block7", Residual_block(gch*4, gch*4, padding=self.padding)),
                            ("res_block8", Residual_block(gch*4, gch*4, padding=self.padding)),
                            ("res_block9", Residual_block(gch*4, gch*4, padding=self.padding)),

                            ("deconv1", Deconvolution2D([2, gch*2, 128, 128], gch*4, gch*2,
                                                        kernel_size=3, stride=2,
                                                        padding_mode="same",
                                                        norm_type="Ins")),

                            ("deconv2", Deconvolution2D([2, gch, 256, 256], gch*2, gch,
                                                        kernel_size=3, stride=2,
                                                        padding_mode="same",
                                                        norm_type="Ins")),
                            ("conv_final", Convolution2D(gch, 1,
                                                        kernel_size=3, stride=1,
                                                        padding_mode="same",
                                                        norm_type=None,
                                                        do_relu=False)),
                            ]))

        self.tanh = nn.Tanh()

    def forward(self, input):

        # Some padding
        padded_input = F.pad(input, (3,3,3,3), mode=self.padding)
        print("Input Shape is: ", padded_input.shape)
        output = self.generator(padded_input)

        if self.skip_conn is True:
            output = self.tanh(input + output)

        else:
            output = self.tanh(output)

        print("Generator_S_T: ", output.shape)
        return output



class Discriminator_T(nn.Module):
    def __init__(self, input_ch):
        super(Discriminator_T, self).__init__()

        self.padding = "constant"
        dch = 64 # Discriminator minimum channel multiple


        self.conv1 = Convolution2D(input_ch, dch,
                                   kernel_size=4, stride=2,
                                   padding_mode="VALID",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv2 = Convolution2D(dch, dch*2,
                                   kernel_size=4, stride=2,
                                   padding_mode="VALID",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv3 = Convolution2D(dch*2, dch*4,
                                   kernel_size=4, stride=2,
                                   padding_mode="VALID",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv4 = Convolution2D(dch*4, dch*8,
                                   kernel_size=4, stride=1,
                                   padding_mode="VALID",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv5 = Convolution2D(dch*8, 1,
                                   kernel_size=4, stride=1,
                                   padding_mode="VALID",
                                   norm_type=None,
                                   do_relu=False)


    def forward(self, input):

        # Some Padding
        padded_input = F.pad(input, (2,2,2,2), mode=self.padding)
        output = self.conv1(padded_input)

        # Some padding
        output = F.pad(output, (2,2,2,2), mode=self.padding)
        output = self.conv2(output)

        # Some padding
        output = F.pad(output, (2,2,2,2), mode=self.padding)
        output = self.conv3(output)

        # Some padding
        output = F.pad(output, (2,2,2,2), mode=self.padding)
        output = self.conv4(output)

        # Some padding
        output = F.pad(output, (2,2,2,2), mode=self.padding)
        output = self.conv5(output)

        return output


if __name__ == "__main__":
    model = Generator_S_T(input_ch = 1, skip_conn = True)
    summary(model, input_size=(1, 256, 256))

    model = Discriminator_T(input_ch = 1)
    summary(model, input_size=(1, 256, 256))