import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler

# Summary/Testing of modules
from torchsummary import summary
import argparse
###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

###############################################################################
# Network Functions
###############################################################################

def define_G(input_nc, output_nc=0, ngf=32, netG='resnet_9blocks', norm='batch', dropout_rate=0.75, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator
    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        dropout_rate(float) -- fraction of dropout : 0.75(default)
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    Our current implementation provides these generators:

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
            Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm) #nn.BatchNorm2d or nn.InstanceNorm2d

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, dropout_rate=0, n_blocks=9, n_downsampling=2, n_upsampling=2)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, dropout_rate=0, n_blocks=6, n_downsampling=2, n_upsampling=2)
    elif netG == 'decoder':
        net = ResnetDecoder(input_nc, output_nc, ngf, norm_layer=norm_layer, dropout_rate=dropout_rate, n_blocks=4, n_downsampling=0, n_upsampling=3)
    elif netG == 'encoder':
        net = ResnetEncoder(input_nc, ngf, norm_layer=norm_layer, dropout_rate=dropout_rate) # fin
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator
    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | aux | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'aux': # default PatchGAN with an auxiliary classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, is_aux=True)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_C(input_nc, num_classes, netC='basic', norm='none', dropout_rate=0.75, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a Pixel Classifier (for Segmentation)
    Parameters:
        input_nc (int) -- the number of channels in input images
        ngf (int) -- the number of filters in the last conv layer
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        dropout_rate(float) -- fraction of dropout : 0.75(default)
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    Our current implementation provides these generators:

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
            Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netC == 'basic':
        net = Classifier(input_nc, num_classes, norm_layer=norm_layer, dropout_rate=dropout_rate)
    elif netC == 'atrous': # Yet to finish
        net = Classifier(input_nc, num_classes, norm_layer=norm_layer, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError('Classifier model name [%s] is not recognized' % netC)
    return init_net(net, init_type, init_gain, gpu_ids)

###############################################################################
# Network Modules
###############################################################################

class Classifier(nn.Module):
    """Basic Convolutional Classifier consisting of a simple convolution and an upscaling block."""

    def __init__(self, input_nc, num_classes=5, norm_layer=nn.Identity, dropout_rate=0.75) :
        super(Classifier, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ngf = num_classes
        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(input_nc, num_classes, kernel_size=1, padding=0, bias=use_bias),
                 nn.Dropout(p=dropout_rate),
                 norm_layer(num_classes),
                 nn.ReLU(False)]

        model += [nn.UpsamplingBilinear2d(size=(256,256))]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetEncoder(nn.Module):
    """Resnet-based encoder that consists of Resnet blocks between a few downsampling, upsampling and occasional maxpool operations.
    We wrote the code to identically resemble the tensorflow encoder network and as given in the paper."""

    def __init__(self, input_nc, ngf=16, norm_layer=nn.BatchNorm2d, dropout_rate=0.75, padding_type='zero'):
        """Construct a Resnet-based encoder
        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            dropout_rate(float) -- fraction of dropout : 0.75(default)
            padding_type(string)-- zero, meaning add '0' as padding to get o/p of same dimension a.k.a 'CONSTANT'
        """

        super(ResnetEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # C16
        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 nn.Dropout(p=dropout_rate),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # R16, M
        model += [ResidualBlock(ngf, ngf, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True),
                  nn.MaxPool2d(kernel_size=2, stride=2, padding=1)]


        # R32, M
        model += [ResidualBlock(ngf, ngf*2, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True),
                  nn.MaxPool2d(kernel_size=2, stride=2, padding=1)]

        # R64_1, R64_2, M
        model += [ResidualBlock(ngf*2, ngf*4, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True),
                  ResidualBlock(ngf*4, ngf*4, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True),
                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]

        # R128_1, R128_2
        model += [ResidualBlock(ngf*4, ngf*8, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True),
                  ResidualBlock(ngf*8, ngf*8, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True)]

        # R256_1, R256_2, R256_3, R256_4
        model += [ResidualBlock(ngf*8, ngf*16, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True),
                  ResidualBlock(ngf*16, ngf*16, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True),
                  ResidualBlock(ngf*16, ngf*16, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True),
                  ResidualBlock(ngf*16, ngf*16, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True)]

        # R512_1, R512_2
        model += [ResidualBlock(ngf*16, ngf*32, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True),
                  ResidualBlock(ngf*32, ngf*32, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True)]

        # D512_1, D512_2
        model += [DilatedResidualBlock(ngf*32, ngf*32, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True),
                  DilatedResidualBlock(ngf*32, ngf*32, padding_type=padding_type, norm_layer=norm_layer,
                                dropout_rate=dropout_rate, use_bias=use_bias),
                  nn.ReLU(True)]

        # C512_1, C512_2
        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf*32, ngf*32, kernel_size=3, padding=0, bias=use_bias),
                  nn.Dropout(p=dropout_rate),
                  norm_layer(ngf*32),
                  nn.ReLU(True),

                  nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf*32, ngf*32, kernel_size=3, padding=0, bias=use_bias),
                  nn.Dropout(p=dropout_rate),
                  norm_layer(ngf*32),
                  nn.ReLU(True)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""

        return self.model(input)

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.BatchNorm2d, dropout_rate=0, n_blocks=6, n_downsampling=2, n_upsampling=2, padding_type='zero'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            dropout_rate(float) -- fraction of dropout : 0.75(default)
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)] # current filters -> 128

        mult = (2 ** n_downsampling) # 2^2 = 4
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResidualBlock(ngf * mult, ngf * mult, padding_type=padding_type, norm_layer=norm_layer, dropout_rate=dropout_rate, use_bias=use_bias),
                      nn.ReLU(True)]

        for i in range(n_upsampling):  # add upsampling layers
            mult = 2 ** (n_upsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        out = input + self.model(input)

        out = F.tanh(out)
        return out

class ResnetDecoder(nn.Module):
    """Resnet-based Decoder that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.BatchNorm2d, dropout_rate=0.75, n_blocks=6, n_downsampling=2, n_upsampling=2, padding_type='zero'):
        """Construct a Resnet-based Decoder
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            dropout_rate(float) -- fraction of dropout : 0.75(default)
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        mult = 4

        model = [nn.Conv2d(input_nc, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
                 norm_layer(ngf * mult),
                 nn.ReLU(True)]

        mult = (4)
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResidualBlock(ngf * mult, ngf * mult, padding_type=padding_type, norm_layer=norm_layer, dropout_rate=dropout_rate, use_bias=use_bias)]

        # 128 -> 64
        model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.ReLU(True)]

        # 64 -> 64
        model += [nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.ReLU(True)]

        mult = int(mult / 2) # (2)

        # 64 -> 32
        model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(int(ngf * mult / 2), output_nc, kernel_size=7, padding=0)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, image):
        """Standard forward"""
        out = image + self.model(input)

        out = F.tanh(out)
        return out

class ResidualBlock(nn.Module):
    """Define a Resnet block with variable channel convolutions"""

    def __init__(self, dim_in, dim_out, padding_type, norm_layer, dropout_rate, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResidualBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.padding_type = padding_type

        self.conv_block = self.build_conv_block(dim_in, dim_out, padding_type, norm_layer, dropout_rate, use_bias)

        self.channel_pad = not (dim_in == dim_out)

    def build_conv_block(self, dim_in, dim_out, padding_type, norm_layer, dropout_rate, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            dropout_rate(float) -- fraction of dropout : 0.75(default)
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1 # Should it be 0?
        else:
            raise NotImplemesntedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=p, bias=use_bias),
                       nn.Dropout(p=dropout_rate),
                       norm_layer(dim_out),
                       nn.ReLU(True)
                      ]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1 # Should it be 0?
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=p, bias=use_bias),
                       nn.Dropout(p=dropout_rate),
                       norm_layer(dim_out)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""

        out = self.conv_block(x)

        if self.channel_pad: # Probably not needed.
            padding = self.padding_type if (self.padding_type != 'zero') else 'constant'
            x = F.pad(x, (0,0,0,0,(self.dim_out-self.dim_in)//2, (self.dim_out-self.dim_in)//2), mode=padding)

        out = x + out  # add skip connections
        return out

class DilatedResidualBlock(nn.Module):
    """Define a Resnet block with variable channel convolutions and additional kernel dilations"""

    def __init__(self, dim_in, dim_out, padding_type, norm_layer, dropout_rate, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(DilatedResidualBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.padding_type = padding_type

        rate = 2 # dilation rate (defaul : 2)

        self.conv_block = self.build_conv_block(dim_in, dim_out, rate, padding_type, norm_layer, dropout_rate, use_bias)

        self.channel_pad = not (dim_in == dim_out)

    def build_conv_block(self, dim_in, dim_out, rate, padding_type, norm_layer, dropout_rate, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            dropout_rate(float) -- fraction of dropout : 0.75(default)
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 2 # dilation padding (normal amount i.e 1 x 2)
        else:
            raise NotImplemesntedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=p, bias=use_bias, dilation=rate),
                       nn.Dropout(p=dropout_rate),
                       norm_layer(dim_out),
                       nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 2 # dilation padding (normal amount i.e 1 x 2)
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=p, bias=use_bias, dilation=rate),
                       nn.Dropout(p=dropout_rate),
                       norm_layer(dim_out)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""

        out = self.conv_block(x)

        if self.channel_pad:
            padding = self.padding_type if (self.padding_type != 'zero') else 'constant'
            x = F.pad(x, (0,0,0,0,(self.dim_out-self.dim_in)//2, (self.dim_out-self.dim_in)//2), mode=padding)

        out = x + out  # add skip connections
        return out

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, is_aux=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 2
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters (Loop count till (n_layers - 1))
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult_prev = nf_mult
        # check after this

        if is_aux:
            sequence += [nn.Conv2d(ndf * nf_mult_prev, 2, kernel_size=kw, stride=1, padding=padw)]  # output 2 channel prediction map
        else:
            sequence += [nn.Conv2d(ndf * nf_mult_prev, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class DICELoss(nn.Module):
    """
    Define Dice Loss Objective.
    """

    def __init__(self, num_classes=5, epsilon=1e-7, p=2):
        """ Initialize the DICELoss class.
        Parameters:
            num_classes (int) - - number of classes for which the loss is calculated
            epsilon -- smoothness considered
        Note: Do not use sigmoid as the last layer of Discriminator.
        """
        super(DICELoss, self).__init__()

        self.softmax = nn.Softmax(dim = 1) # dim 1 --> num_filters/classes
        self.num_classes=num_classes
        self.eps = epsilon
        self.p = p

    def __call__(self, mask, gt):
        """Calculate dice loss given Classifier's output and ground trutg labels.
        Parameters:
            mask (tensor) : The segmentation prediction output from the classifier
            ground truth (gt) : The actual segmentation groundtruth
        Returns:
            The dice score loss
        """
        dice = 0
        mask = self.softmax(mask)
        for i in range(num_classes):
            prediction = mask[:, i, :, :]
            target = gt[:, i, :, :]
            inse = torch.sum(prediction * target)

            l = torch.sum(prediction.pow(self.p))
            r = torch.sum(target.pow(self.p))
            dice += 2.0 * inse/(l+r+self.eps)

        return 1 - 1.0 * dice / self.num_classes

class WCELoss(nn.Module):
    """
    Define Weighted Cross Entropy Loss Objective.
    """

    def __init__(self, num_classes=5):
        """ Initialize the Weighted CELoss class.
        Parameters:
            num_classes (int) -- number of classes for calculating the weight of the CE loss.
        """
        super(WCELoss, self).__init__()

        self.num_classes = num_classes

    def __call__(self, logits, gt):
        weights = []
        for i in range(self.num_classes):
            gti = gt[:, i, :, :]
            weights[i] = 1 - (torch.sum(gti)/torch.sum(gt))
        self.loss = nn.CrossEntropyLoss(weight=weights)

        return self.loss(logits, gt)


###############################################################################
# Summary Executable Functions
###############################################################################

class switch(object):
    value = None
    def __new__(class_, value):
        class_.value = value
        return True

def case(*args):
    return any((arg == switch.value for arg in args))

class NetTester():

    def __init__(self):
        """Consider this as a temporary specific network tester.
        I will make this an abstract class, so that one can add their torchsummary options here.
        Parameters:
            variables (if needed)
            options     --      Switch case scenario (to choose what network to test)
        """
        options = ['encoder',
           'segmentor',
           'generator',
           'decoder',
           'discriminatorT',
           'discriminatorS',
           'discriminatorP'
           ]

        self.dropout_rate = 0.75
        self.init_type = 'normal'
        self.init_gain = 0.01
        self.gpu_ids = ''

        self.num_classes=5

        print ("Welcome to the Network Tester :")
        print (options)

    def parse(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--network', required=True, help='choose from the networks above. We will summarize the network for you')

        return parser.parse_args()

    def execute(self, n):
        while switch(n):
            if case('encoder'):
                netE = define_G(input_nc=1,
                                ngf=16, netG='encoder',
                                norm = 'batch', dropout_rate=self.dropout_rate,
                                init_type=self.init_type, init_gain=self.init_gain,
                                gpu_ids=self.gpu_ids) # a-OK
                summary(netE, input_size=(1, 256, 256))
                break

            if case('segmentor'):
                netC = define_C(input_nc=512, num_classes=self.num_classes,
                                netC='basic',
                                norm='none', dropout_rate=self.dropout_rate,
                                init_type=self.init_type, init_gain=0.01,
                                gpu_ids=self.gpu_ids) # a-OK

                summary(netC, input_size=(512, 32, 32))
                break

            if case('generator'):
                netG_T = define_G(input_nc=1, output_nc=1,
                                  ngf=32, netG='resnet_9blocks',
                                  norm='instance', dropout_rate=self.dropout_rate,
                                  init_type=self.init_type, init_gain=self.init_gain,
                                  gpu_ids=self.gpu_ids) # validate dimensions

                summary(netG_T, input_size=(1, 256, 256))
                break

            if case('decoder'):
                netU = define_G(input_nc=512, output_nc=1,
                                ngf=32, netG='decoder',
                                norm='instance', dropout_rate=self.dropout_rate,
                                init_type=self.init_type, init_gain=self.init_gain,
                                gpu_ids=self.gpu_ids) # validate dimensions

                summary(netU, input_size=[(512, 32, 32), (1, 256, 256)])
                break

            if case('discriminatorT'):
                netD_T = define_D(input_nc=1, ndf=64,
                                  netD='basic',
                                  norm='instance',
                                  init_type=self.init_type, init_gain=self.init_gain,
                                  gpu_ids=self.gpu_ids) # a-OK

                summary(netD_T, input_size=[(1, 256, 256)])
                break

            if case('discriminatorS'):
                netD_S = define_D(input_nc=1, ndf=64,
                                  netD='aux',
                                  norm='instance',
                                  init_type=self.init_type, init_gain=self.init_gain,
                                  gpu_ids=self.gpu_ids) # a-OK

                summary(netD_S, input_size=[(1, 256, 256)])
                break

            if case('discriminatorP'):
                netD_P = define_D(input_nc=5, ndf=64,
                                  netD='basic',
                                  norm='instance', init_type=self.init_type, init_gain=self.init_gain,
                                  gpu_ids=self.gpu_ids) # a-OK

                summary(netD_P, input_size=[(5, 256, 256)])
                break

            print ("Only above options are allowed.")
            break

if __name__ == "__main__":

    nt = NetTester()
    test = nt.parse()
    nt.execute(test.network)
