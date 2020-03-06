import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image, to_tensor

from collections import OrderedDict

from torchsummary import summary

from layers import Convolution2D, DilatedConv2D, Deconvolution2D
from modules import Residual_block, Dilated_Residual_block

import json

'''
with open('./config_param.json') as config_file:
    config = json.load(config_file)

BATCH_SIZE = int(config['batch_size'])
POOL_SIZE = int(config['pool_size'])
'''
# The height of each image.
IMG_HEIGHT = 256

# The width of each image.
IMG_WIDTH = 256


class SIFA_generators(nn.Module):

    def __init__(self, skip_conn=False, is_training=True, dropout_rate=0.75):
        super(SIFA_generators, self).__init__()

        self.skip_conn = skip_conn
        self.dropout_rate = dropout_rate
        self.is_training = is_training

        self.generator_s_t = Generator_S_T(input_ch=3, skip_conn=self.skip_conn)
        # input_ch could be either 3 or 1

        self.encoder = Encoder(input_ch=3, skip_conn=self.skip_conn, dropout_rate=self.dropout_rate, is_training=self.is_training)
        # input_ch could be either 3 or 1

        self.decoder = Decoder(latent_inp_ch=512, skip_conn=self.skip_conn)

        self.segmenter = Segmenter(latent_inp_ch=512, dropout_rate=self.dropout_rate)


    def forward(self, inputs):
        # images_s, images_t
        images_s = inputs['images_s'].float()
        images_t = inputs['images_t'].float()

        print("Images s & t", images_s.shape, images_t.shape)
        # ---------- GRAPH 1 GENERATORS---------------------------------------

        fake_images_t = self.generator_s_t(torch.cat((images_s, images_s, images_s), 1))
        print("fake_images_t", fake_images_t.shape)
        latent_fake_t = self.encoder(torch.cat((fake_images_t, fake_images_t, fake_images_t), 1))

        pred_mask_fake_t = self.segmenter(latent_fake_t)
        cycle_images_s = self.decoder(latent_fake_t, fake_images_t)

        # ---------- GRAPH 2 GENERATORS---------------------------------------

        latent_t = self.encoder(torch.cat((images_t, images_t, images_t), 1))

        pred_mask_t = self.segmenter(latent_t)
        fake_images_s = self.decoder(latent_t, images_t.unsqueeze(1))

        cycle_images_t = self.generator_s_t(torch.cat((fake_images_s, fake_images_s, fake_images_s), 1))




        return {
            'fake_images_s': fake_images_s,
            'fake_images_t': fake_images_t,
            'cycle_images_s': cycle_images_s,
            'cycle_images_t': cycle_images_t,
            'pred_mask_s': pred_mask_t,
            'pred_mask_t': pred_mask_t,
            'pred_mask_fake_s': pred_mask_fake_t,
            'pred_mask_fake_t': pred_mask_fake_t,
        }


class SIFA_discriminators(nn.Module):

    def __init__(self):
        super(SIFA_discriminators, self).__init__()

        self.discriminator_t = Discriminator_T(input_ch=1)

        self.discriminator_aux = Discriminator_AUX(input_ch=1)

        self.discriminator_mask = Discriminator_MASK(input_ch=5)


    def forward(self, inputs):
        # images_s, images_t, fake_images_s, fake_images_t, fake_pool_s, fake_pool_t, cycle_images_s, pred_mask_fake_t, pred_mask_t
        images_s = inputs['images_s'].float()
        images_t = inputs['images_t'].float()
        fake_images_s = input['fake_images_s'].float()
        fake_images_t = input['fake_images_t'].float()
        fake_pool_s = inputs['fake_pool_s'].float()
        fake_pool_t = inputs['fake_pool_t'].float()
        cycle_images_s = inputs['cycle_images_s'].float()
        pred_mask_fake_t = inputs['pred_mask_fake_t'].float()
        pred_mask_t = inputs['pred_mask_t'].float()

        images_s_dd = images_s.detach()
        images_t_dd = images_t.detach()
        prob_real_s_is_real, prob_real_s_aux = self.discriminator_aux(images_s_dd[:, 1, :, :].unsqueeze(1))
        prob_real_t_is_real = self.discriminator_t(images_t_dd[:, 1, :, :].unsqueeze(1))

        prob_fake_s_is_real, prob_fake_s_aux_is_real = self.discriminator_aux(fake_images_s)
        prob_fake_t_is_real = self.discriminator_t(fake_images_t)

        fake_pool_s_dd = fake_pool_s.detach()
        fake_pool_t_dd = fake_pool_t.detach()
        prob_fake_pool_s_is_real, prob_fake_pool_s_aux_is_real = self.discriminator_aux(fake_pool_s_dd)
        prob_fake_pool_t_is_real = self.discriminator_t(fake_pool_t_dd)

        cycle_images_s_dd = cycle_images_s.detach()
        prob_cycle_s_is_real, prob_cycle_s_aux_is_real = self.discriminator_aux(cycle_images_s_dd)

        pred_mask_fake_t_dd = pred_mask_fake_t.detach()
        prob_pred_mask_fake_t_is_real = self.discriminator_mask(pred_mask_fake_t_dd)
        prob_pred_mask_t_is_real = self.discriminator_mask(pred_mask_t)

        return {
            'prob_real_s_is_real': prob_real_s_is_real,
            'prob_real_t_is_real': prob_real_t_is_real,
            'prob_fake_s_is_real': prob_fake_s_is_real,
            'prob_fake_t_is_real': prob_fake_t_is_real,
            'prob_fake_pool_s_is_real': prob_fake_pool_s_is_real,
            'prob_fake_pool_t_is_real': prob_fake_pool_t_is_real,
            'prob_pred_mask_fake_t_is_real': prob_pred_mask_fake_t_is_real,
            'prob_pred_mask_t_is_real': prob_pred_mask_t_is_real,
            'prob_fake_s_aux_is_real': prob_fake_s_aux_is_real,
            'prob_fake_pool_s_aux_is_real': prob_fake_pool_s_aux_is_real,
            'prob_cycle_s_aux_is_real': prob_cycle_s_aux_is_real,
        }





class Generator_S_T(nn.Module):

    def __init__(self, input_ch, skip_conn=False):
        super(Generator_S_T, self).__init__()

        gch = 32 # generator minimum channel multiple
        self.padding = "constant"
        self.skip_conn = skip_conn

        self.generator = nn.Sequential(OrderedDict([
                            ("conv1", Convolution2D(input_ch, gch,
                                                    kernel_size=7, stride=1,
                                                    deviation=0.02,
                                                    norm_type="Ins")),
                            ("conv2", Convolution2D(gch, gch*2,
                                                    kernel_size=3, stride=2,
                                                    deviation=0.02,
                                                    padding_mode="same",
                                                    norm_type="Ins")),
                            ("conv3", Convolution2D(gch*2, gch*4,
                                                    kernel_size=3, stride=2,
                                                    deviation=0.02,
                                                    padding_mode="same",
                                                    norm_type="Ins")),

                            ("res_block1", Residual_block(gch*4, gch*4, padding=self.padding, norm_type="Ins", deviation=0.02)),
                            ("res_block2", Residual_block(gch*4, gch*4, padding=self.padding, norm_type="Ins", deviation=0.02)),
                            ("res_block3", Residual_block(gch*4, gch*4, padding=self.padding, norm_type="Ins", deviation=0.02)),
                            ("res_block4", Residual_block(gch*4, gch*4, padding=self.padding, norm_type="Ins", deviation=0.02)),
                            ("res_block5", Residual_block(gch*4, gch*4, padding=self.padding, norm_type="Ins", deviation=0.02)),
                            ("res_block6", Residual_block(gch*4, gch*4, padding=self.padding, norm_type="Ins", deviation=0.02)),
                            ("res_block7", Residual_block(gch*4, gch*4, padding=self.padding, norm_type="Ins", deviation=0.02)),
                            ("res_block8", Residual_block(gch*4, gch*4, padding=self.padding, norm_type="Ins", deviation=0.02)),
                            ("res_block9", Residual_block(gch*4, gch*4, padding=self.padding, norm_type="Ins", deviation=0.02)),

                            ("deconv1", Deconvolution2D([2, gch*2, 128, 128], gch*4, gch*2,
                                                        kernel_size=3, stride=2,
                                                        deviation=0.02,
                                                        padding_mode="same",
                                                        norm_type="Ins")),

                            ("deconv2", Deconvolution2D([2, gch, 256, 256], gch*2, gch,
                                                        kernel_size=3, stride=2,
                                                        deviation=0.02,
                                                        padding_mode="same",
                                                        norm_type="Ins")),
                            ("conv_final", Convolution2D(gch, 1,
                                                        kernel_size=3, stride=1,
                                                        deviation=0.02,
                                                        padding_mode="same",
                                                        norm_type=None,
                                                        do_relu=False)),
                            ]))

        self.tanh = nn.Tanh()

    def forward(self, input):

        # Some padding
        padded_input = F.pad(input, (3,3,3,3), mode=self.padding)

        # print("Input Shape is: ", padded_input.shape)

        output = self.generator(padded_input)
        # print("output Shape is: ", output.shape)
        # print("actual input shape", torch.unsqueeze(input[:,1,:,:],1).shape)

        if self.skip_conn is True:
            output = self.tanh(input[:,1,:,:].unsqueeze(1) + output)

        else:
            output = self.tanh(output)

        # print("Generator_S_T: ", output.shape)
        return output



class Discriminator_T(nn.Module):
    def __init__(self, input_ch):
        super(Discriminator_T, self).__init__()

        self.padding = "constant"
        dch = 64 # Discriminator minimum channel multiple


        self.conv1 = Convolution2D(input_ch, dch,
                                   kernel_size=4, stride=2,
                                   deviation=0.02,
                                   padding_mode="valid",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv2 = Convolution2D(dch, dch*2,
                                   kernel_size=4, stride=2,
                                   deviation=0.02,
                                   padding_mode="valid",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv3 = Convolution2D(dch*2, dch*4,
                                   kernel_size=4, stride=2,
                                   deviation=0.02,
                                   padding_mode="valid",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv4 = Convolution2D(dch*4, dch*8,
                                   kernel_size=4, stride=1,
                                   deviation=0.02,
                                   padding_mode="valid",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv5 = Convolution2D(dch*8, 1,
                                   kernel_size=4, stride=1,
                                   deviation=0.02,
                                   padding_mode="valid",
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

#___________________________________________________________________________________________

class Discriminator_AUX(nn.Module):
    def __init__(self, input_ch):
        super(Discriminator_AUX, self).__init__()

        self.padding = "constant"
        dch = 64 # Discriminator minimum channel multiple


        self.conv1 = Convolution2D(input_ch, dch,
                                   kernel_size=4, stride=2,
                                   deviation=0.02,
                                   padding_mode="valid",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv2 = Convolution2D(dch, dch*2,
                                   kernel_size=4, stride=2,
                                   deviation=0.02,
                                   padding_mode="valid",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv3 = Convolution2D(dch*2, dch*4,
                                   kernel_size=4, stride=2,
                                   deviation=0.02,
                                   padding_mode="valid",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv4 = Convolution2D(dch*4, dch*8,
                                   kernel_size=4, stride=1,
                                   deviation=0.02,
                                   padding_mode="valid",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv5 = Convolution2D(dch*8, 2,
                                   kernel_size=4, stride=1,
                                   deviation=0.02,
                                   padding_mode="valid",
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

        return output[:,0,:,:], output[:,1,:,:]

#___________________________________________________________________________________________

class Discriminator_MASK(nn.Module):
    def __init__(self, input_ch):
        super(Discriminator_MASK, self).__init__()

        self.padding = "constant"
        dch = 64 # Discriminator minimum channel multiple


        self.conv1 = Convolution2D(input_ch, dch,
                                   kernel_size=4, stride=2,
                                   deviation=0.02,
                                   padding_mode="valid",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv2 = Convolution2D(dch, dch*2,
                                   kernel_size=4, stride=2,
                                   deviation=0.02,
                                   padding_mode="valid",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv3 = Convolution2D(dch*2, dch*4,
                                   kernel_size=4, stride=2,
                                   deviation=0.02,
                                   padding_mode="valid",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv4 = Convolution2D(dch*4, dch*8,
                                   kernel_size=4, stride=1,
                                   deviation=0.02,
                                   padding_mode="valid",
                                   norm_type="Ins",
                                   do_relu=True, relu_factor=0.2)

        self.conv5 = Convolution2D(dch*8, 1,
                                   kernel_size=4, stride=1,
                                   deviation=0.02,
                                   padding_mode="valid",
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

#___________________________________________________________________________________________


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
                                ("M3", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),


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

        return dilation_output

#___________________________________________________________________________________________

class Decoder(nn.Module):
    def __init__(self, latent_inp_ch, skip_conn=False):
        super(Decoder, self).__init__()

        dch = 32 # Encoder minimum channel multiple
        self.padding = "constant"
        self.skip_conn = skip_conn

        self.decode_net = nn.Sequential(OrderedDict([

                                ("conv1", Convolution2D(latent_inp_ch, dch*4,
                                                        kernel_size=3, stride=1,
                                                        deviation=0.02,
                                                        padding_mode="same",
                                                        norm_type="Ins")),

                                ("res_block1", Residual_block(dch*4, dch*4, padding=self.padding, norm_type="Ins")),
                                ("res_block2", Residual_block(dch*4, dch*4, padding=self.padding, norm_type="Ins")),
                                ("res_block3", Residual_block(dch*4, dch*4, padding=self.padding, norm_type="Ins")),
                                ("res_block4", Residual_block(dch*4, dch*4, padding=self.padding, norm_type="Ins")),

                                ("deconv1", Deconvolution2D([2, dch*2, 64, 64], dch*4, dch*2,
                                                            kernel_size=3, stride=2,
                                                            deviation=0.02,
                                                            padding_mode="same",
                                                            norm_type="Ins")),

                                ("deconv2", Deconvolution2D([2, dch*2, 128, 128], dch*2, dch*2,
                                                            kernel_size=3, stride=2,
                                                            deviation=0.02,
                                                            padding_mode="same",
                                                            norm_type="Ins")),

                                ("deconv3", Deconvolution2D([2, dch, 256, 256], dch*2, dch,
                                                            kernel_size=3, stride=2,
                                                            deviation=0.02,
                                                            padding_mode="same",
                                                            norm_type="Ins")),

                                ("conv_final", Convolution2D(dch, 1,
                                                        kernel_size=3, stride=1,
                                                        deviation=0.02,
                                                        padding_mode="same",
                                                        norm_type=None,
                                                        do_relu=False))
                                ]))

        self.tanh = nn.Tanh()


    def forward(self, latent_input, image_input):

        output = self.decode_net(latent_input)

        if self.skip_conn is True:
            output = self.tanh(image_input + output)

        else:
            ouptut = self.tanh(output)

        return output

#___________________________________________________________________________________________

class Segmenter(nn.Module):
    def __init__(self, latent_inp_ch, dropout_rate=0.75):
        super(Segmenter, self).__init__()

        self.segment_net = Convolution2D(latent_inp_ch, 5,
                                       kernel_size=1, stride=1,
                                       padding_mode="same",
                                       norm_type=None,
                                       do_relu=False,
                                       dropout_rate=dropout_rate)

        self.upsample = torch.nn.UpsamplingBilinear2d(size=(256,256))



    def forward(self, latent_input):

        output = self.segment_net(latent_input)
        output = self.upsample(output)

        # resized_output = to_tensor(resize(to_pil_image(output.squeeze())))

        return output

# if __name__ == "__main__":
    # model = Segmenter(latent_inp_ch = 512, dropout_rate=0.75)
    # summary(model, input_size=(512, 40, 40))
    #
    # model = Encoder(input_ch = 1, skip_conn = True)
    # summary(model, input_size=(1, 256, 256))
    #
    # model = Decoder(latent_inp_ch = 512, skip_conn = True)
    # summary(model, input_size=[(512, 32, 32), (1, 256, 256)])
    #
    # model = Generator_S_T(input_ch = 1, skip_conn = True)
    # summary(model, input_size=(1, 256, 256))
    #
    # model = Discriminator_T(input_ch = 1)
    # summary(model, input_size=(1, 256, 256))
    #
    # model = Discriminator_AUX(input_ch = 1)
    # summary(model, input_size=(1, 256, 256))

    # model = SIFA_generators(skip_conn=False, is_training=True, dropout_rate=0.75)
    # summary(model, input_size=[(3, 256, 256), (3, 256, 256)])

    # model = SIFA_discriminators()
    # summary(model, input_size=[(3, 256, 256), (3, 256, 256), (1,256,256), (1, 256, 256), (1, 256, 256), (1, 256, 256), (1, 256, 256), (5,256,256), (5,256,256)])
