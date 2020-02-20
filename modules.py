import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict

from torchsummary import summary

from layers import Convolution2D, Deconvolution2D, DilatedConv2D

class Residual_block(nn.Module):
    def __init__(self, input_ch, output_ch, padding="reflect"):
        super(Residual_block, self).__init__()

        self.padding=padding

        self.conv1 = Convolution2D(input_ch, output_ch, kernel_size=3, stride=1, padding_mode="valid", norm_type="Ins")
        self.conv2 = Convolution2D(output_ch, output_ch, kernel_size=3, stride=1,padding_mode="valid", do_relu=False, norm_type="Ins")

        self.skip = torch.nn.ReLU()

    def forward(self, input):

        # Some padding
        padded_input = F.pad(input, (1,1,1,1), mode=self.padding)
        output = self.conv1(padded_input)

        # Some padding
        padded_output = F.pad(output, (1,1,1,1), mode=self.padding)
        output = self.conv2(padded_output)

        output = self.skip(output+input)

        print("Residual_block: ", output.shape)
        return output

if __name__ == "__main__":
    model = Residual_block(input_ch=64, output_ch=64)
    summary(model, input_size=(64, 256, 256))
