import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict

from torchsummary import summary

from layers import Convolution2D, Deconvolution2D, DilatedConv2D

class Residual_block(nn.Module):
    def __init__(self, input_ch, output_ch,
                 padding="reflect",
                 norm_type=None,
                 dropout_rate=0.75,
                 is_training=True,
                 deviation=None,
                 channel_pad=False):
        super(Residual_block, self).__init__()

        self.input_ch = input_ch
        self.output_ch = output_ch

        self.padding = padding

        self.deviation = 0.01 if deviation==None else deviation
        self.channel_pad = channel_pad # If True, we pad channels with (output_ch-input_ch/2)

        self.conv1 = Convolution2D(input_ch, output_ch,
                                   kernel_size=3, stride=1,
                                   deviation=self.deviation,
                                   padding_mode="valid",
                                   norm_type=norm_type,
                                   dropout_rate=dropout_rate,
                                   is_training=is_training)

        self.conv2 = Convolution2D(output_ch, output_ch,
                                   kernel_size=3, stride=1,
                                   deviation=self.deviation,
                                   padding_mode="valid",
                                   do_relu=False,
                                   norm_type=norm_type,
                                   dropout_rate=dropout_rate,
                                   is_training=is_training)

        self.skip = torch.nn.ReLU()

    def forward(self, input):

        # Some padding
        padded_input = F.pad(input, (1,1,1,1), mode=self.padding)
        output = self.conv1(padded_input)

        # Some padding
        padded_output = F.pad(output, (1,1,1,1), mode=self.padding)
        output = self.conv2(padded_output)

        # If True, we pad channels with (output_ch-input_ch/2)
        if self.channel_pad:
            # print("Padded channel number: ", input.shape)
            # print("Padding Channels: ")
            input = F.pad(input,
                           (0,0,0,0,(self.output_ch-self.input_ch)//2, (self.output_ch-self.input_ch)//2),
                           mode=self.padding)
            # print("Padded channel number: ", input.shape)

        output = self.skip(output+input)

        # print("Residual_block: ", output.shape)
        return output

class Dilated_Residual_block(nn.Module):
    def __init__(self, input_ch, output_ch,
                 padding="reflect",
                 norm_type=None,
                 dropout_rate=0.75,
                 is_training=True,
                 deviation=None):
        super(Dilated_Residual_block, self).__init__()

        self.padding = padding

        self.deviation = 0.01 if deviation==None else deviation

        self.dilated_conv1 = DilatedConv2D(input_ch, output_ch,
                                          kernel_size=3,
                                          rate=2,
                                          deviation=self.deviation,
                                          padding_mode="valid",
                                          norm_type=norm_type,
                                          dropout_rate=dropout_rate,
                                          is_training=is_training)

        self.dilated_conv2 = DilatedConv2D(output_ch, output_ch,
                                           kernel_size=3,
                                           rate=2,
                                           deviation=self.deviation,
                                           padding_mode="valid",
                                           do_relu=False,
                                           norm_type=norm_type,
                                           dropout_rate=dropout_rate,
                                           is_training=is_training)

        self.skip = torch.nn.ReLU()

    def forward(self, input):

        # Some padding
        padded_input = F.pad(input, (2,2,2,2), mode=self.padding)
        output = self.dilated_conv1(padded_input)

        # Some padding
        output = F.pad(output, (2,2,2,2), mode=self.padding)
        output = self.dilated_conv2(output)

        output = self.skip(output+input)

        return output


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        # modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)




if __name__ == "__main__":
    model = Residual_block(input_ch=64, output_ch=64)
    summary(model, input_size=(64, 256, 256))

    model = Dilated_Residual_block(input_ch=64, output_ch=64)
    summary(model, input_size=(64, 256, 256))
