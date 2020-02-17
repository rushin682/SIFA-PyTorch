import torch
import torch.nn.functional as F
import torch.nn as nn

class Convolution2D(nn.module):
    def __init__(self,
                input_ch, output_ch,
                kernel_size=3, stride=1,
                padding_mode="VALID",
                dropout_rate=None,
                norm_type=None,
                do_relu=True, relu_factor=0,
                is_training=True):

        super(Convolution2D, self).__init__()

        self.do_relu = do_relu
        self.norm_type = norm_type

        padding = 0 if padding_mode="VALID" else 1

        self.conv = nn.Conv2d(input_ch, output_ch, kernel_size, stride, padding)
        self.dropout = nn.Dropout(p=dropout_rate)

        if norm_type=='Ins':
            self.norm = nn.InstanceNorm2d(output_ch)

        elif norm_type=='Batch':
            self.norm = nn.BatchNorm2d(output_ch, momentum=0.9)

        self.relu = nn.LeakyReLU(negative_slope = relu_factor)

    def forward(self, input):

        output = self.conv(input)
        output = self.dropout(output)

        if not self.norm_type is None:
            output = self.norm(output)

        if self.do_relu:
            output = self.relu(output)

        return output

        
