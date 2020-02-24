import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage, Resize

from collections import OrderedDict

from torchsummary import summary

from layers import Convolution2D, DilatedConv2D, Deconvolution2D
from modules import Residual_block, Dilated_Residual_block
import json

class Segmenter(nn.Module):
    def __init__(self, latent_inp_ch, dropout_rate=0.75):
        super(Segmenter, self).__init__()

        self.segmenter = Convolution2D(latent_inp_ch, 5,
                                       kernel_size=1, stride=1,
                                       padding_mode="same",
                                       norm_type=None,
                                       do_relu=False,
                                       dropout_rate=dropout_rate)

        # self.toPIL = ToPILImage()
        #
        # self.resize = Resize((256,256))
        #
        # self.toTensor = ToTensor()


    def forward(self, latent_input):

        output = self.segmenter(latent_input)

        # resized_output = self.toTensor(self.resize(self.toPIL(output)))

        return output



if __name__ == "__main__":
    model = Segmenter(latent_inp_ch = 512, dropout_rate=0.75)
    summary(model, input_size=(512, 40, 40))
