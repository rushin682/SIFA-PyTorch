import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Convolution2D, DilatedConv2D, Deconvolution2D
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

    def __init__(self, dropout_rate):
        super(Generator_S_T, self).__init__()

        self.generator = nn.Sequential(OrderedDict([
                            ('conv1', Convolution2D(input_ch, 32,
                                                    kernel_size=7, stride=1,
                                                    norm_type='Ins')),
                            ('conv2', Convolution2D(32, 32*2,
                                                    kernel_size=3, stride=2,
                                                    norm_type='Ins')),
                            ('conv3', Convolution2D(32*2, 32*4,
                                                    kernel_size=3, stride=2,
                                                    norm_type='Ins')),

                            ('')
        ]))
