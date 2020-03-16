import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict

from torchsummary import summary

class Convolution2D(nn.Module):
    def __init__(self,
                input_ch, output_ch,
                kernel_size=7, stride=1,
                deviation=0.01,
                padding_mode="valid",
                dropout_rate=0,
                norm_type=None,
                do_relu=True, relu_factor=0,
                is_training=True):

        super(Convolution2D, self).__init__()

        self.do_relu = do_relu
        self.norm_type = norm_type

        padding = 0 if padding_mode=="valid" else 1

        self.conv = nn.Conv2d(input_ch, output_ch, kernel_size, stride, padding=padding)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Weight Initialization --> Conv
        nn.init.normal_(self.conv.weight.data, std=deviation)

        if norm_type == "Ins":
            self.norm = nn.InstanceNorm2d(output_ch)

        elif norm_type == "Batch":
            self.norm = nn.BatchNorm2d(output_ch, momentum=0.9, track_running_stats=is_training)
            nn.init.normal_(self.norm.weight.data, 1.0, deviation)
            nn.init.constant_(self.norm.bias.data, 0.0)

        self.relu = nn.LeakyReLU(negative_slope=relu_factor)

        
        
        nn.init.normal_


    def forward(self, input):

        output = self.conv(input)
        output = self.dropout(output)

        if not self.norm_type is None:
            output = self.norm(output)

        if self.do_relu:
            output = self.relu(output)

        # print("Conv2D: ", output.shape)
        return output


class DilatedConv2D(nn.Module):
    def __init__(self,
                input_ch, output_ch,
                kernel_size=7,
                rate=2,
                deviation=0.01,
                padding_mode="valid",
                dropout_rate=0,
                norm_type=None,
                do_relu=True, relu_factor=0,
                is_training=True):

        super(DilatedConv2D, self).__init__()

        self.do_relu = do_relu
        self.norm_type = norm_type

        padding = 0 if padding_mode=="valid" else 1

        self.dil_conv = nn.Conv2d(input_ch, output_ch, kernel_size, padding=padding, dilation=rate)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Weight Initialization --> Conv
        nn.init.normal_(self.dil_conv.weight.data, std=deviation)

        if norm_type == "Ins":
            self.norm = nn.InstanceNorm2d(output_ch)

        elif norm_type == "Batch":
            self.norm = nn.BatchNorm2d(output_ch, momentum=0.9, track_running_stats=is_training)
            nn.init.normal_(self.norm.weight.data, 1.0, deviation)
            nn.init.constant_(self.norm.bias.data, 0.0)

        self.relu = nn.LeakyReLU(negative_slope=relu_factor)
      


    def forward(self, input):

        output = self.dil_conv(input)
        output = self.dropout(output)

        if not self.norm_type is None:
            output = self.norm(output)

        if self.do_relu:
            output = self.relu(output)

        # print("DilatedConv2D: ", output.shape)
        return output


class Deconvolution2D(nn.Module):
    def __init__(self,
                outshape, input_ch, output_ch=64,
                kernel_size=7, stride=1,
                deviation=0.01,
                padding_mode="valid",
                norm_type=None,
                do_relu=True, relu_factor=0,
                is_training=True):

        super(Deconvolution2D, self).__init__()

        self.outshape = outshape
        self.do_relu = do_relu
        self.norm_type = norm_type

        padding = 0 if padding_mode=="valid" else 1

        self.deconv = nn.ConvTranspose2d(input_ch, output_ch, kernel_size, stride, padding=padding)
        
        # Weight Initialization --> Conv
        nn.init.normal_(self.deconv.weight.data, std=deviation)

        if norm_type == "Ins":
            self.norm = nn.InstanceNorm2d(output_ch)

        elif norm_type == "Batch":
            self.norm = nn.BatchNorm2d(output_ch, momentum=0.9, track_running_stats=is_training)
            nn.init.normal_(self.norm.weight.data, 1.0, deviation)
            nn.init.constant_(self.norm.bias.data, 0.0)

        self.relu = nn.LeakyReLU(negative_slope=relu_factor)

    def forward(self, input):

        output = self.deconv(input, output_size = self.outshape)

        if not self.norm_type is None:
            output = self.norm(output)

        if self.do_relu:
            output = self.relu(output)

        # print("Deconvolution2D: ", output.shape)
        return output


if __name__ == "__main__":
    model = Convolution2D(input_ch=3, output_ch=64, norm_type="Ins", dropout_rate=0.2)
    summary(model, input_size=(3, 256, 256))

    model_dilated_conv = DilatedConv2D(input_ch=3, output_ch=64, norm_type="Ins", dropout_rate=0.2, rate=2)
    summary(model_dilated_conv, input_size=(3, 256, 256))

    model_deconv = Deconvolution2D([2,64,128,128], input_ch=128, output_ch=64, kernel_size=3, stride=2, padding_mode="same", norm_type="Ins")
    summary(model_deconv, input_size=(128, 64, 64))
