import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SIFA(nn.module):
    def __init__(self,)

        self.generator_s_t = Generator_S_T()


class Generator_S_T(nn.module):

    def __init__(self, skip_conn=False):
        super(Generator_S_T, self).__init__()

        gch = 32 # generator minimum channel multiple
        self.padding = "CONSTANT"

        self.generator = nn.Sequential(OrderedDict([
                            ('conv1', Convolution2D(input_ch, gch,
                                                    kernel_size=7, stride=1,
                                                    norm_type="Ins")),
                            ('conv2', Convolution2D(gch, gch*2,
                                                    kernel_size=3, stride=2,
                                                    norm_type="Ins")),
                            ('conv3', Convolution2D(gch*2, gch*4,
                                                    kernel_size=3, stride=2,
                                                    norm_type="Ins")),

                            ('res_block1', Residual_block(gch*4, gch*4, padding=self.padding)),
                            ('res_block2', Residual_block(gch*4, gch*4, padding=self.padding)),
                            ('res_block3', Residual_block(gch*4, gch*4, padding=self.padding)),
                            ('res_block4', Residual_block(gch*4, gch*4, padding=self.padding)),
                            ('res_block5', Residual_block(gch*4, gch*4, padding=self.padding)),
                            ('res_block6', Residual_block(gch*4, gch*4, padding=self.padding)),
                            ('res_block7', Residual_block(gch*4, gch*4, padding=self.padding)),
                            ('res_block8', Residual_block(gch*4, gch*4, padding=self.padding)),
                            ('res_block9', Residual_block(gch*4, gch*4, padding=self.padding)),

                            ('deconv1', Deconvolution2D(gch*4, gch*2,
                                                        kernel_size=3, stride=2,
                                                        padding_mode="SAME",
                                                        norm_type='Ins')),

                            ('deconv2', Deconvolution2D(gch*2, gch,
                                                        kernel_size=3, stride=2,
                                                        padding_mode="SAME",
                                                        norm_type='Ins')),

                            ('conv_final', Convolution2D(gch, 1,
                                                        kernel_size=7, stride=1,
                                                        padding_mode="SAME",
                                                        norm_type=None,
                                                        do_relu=False)),
                            ]))

        self.tanh = nn.Tanh()

    def forward(self, input):

        #Some padding
        output = self.generator(padded_input)

        if skip_conn is True:
            output = self.tanh(input + output)

        else:
            output = self.tanh(output)

        return output

    
