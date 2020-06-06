import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class SIFA(BaseModel):
    """
    Synergistic Image and Feature Adaptation:
        Towards Cross-Modality Domain Adaptation for Medical Image Segmentation
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For SIFA, in addition to GAN losses, we introduce lambda_A, and lambda_B for the following losses.
        A (source domain), B (target domain).
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_S', type=float, default=10.0, help='weight for cycle loss (S -> T -> S)')
            parser.add_argument('--lambda_T', type=float, default=10.0, help='weight for cycle loss (T -> S -> T)')

        return parser

    def __init__(self, opt):
        """Initialize the SIFA class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_T', 'G_T', 'cycle_S', 'D_S', 'G_S', 'cycle_T', 'Seg', 'D_P']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_S', 'fake_T', 'rec_S']
        visual_names_B = ['real_T', 'fake_S', 'rec_T']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_T', 'E', 'U', 'D_T', 'D_S', 'C', 'D_P']
        else:  # during test time, only load Gs
            self.model_names = ['E', 'C']

        # define networks (both Generators and discriminators)
        # The naming is similar to those used in the paper.
        """
        G_T : Generator T (generates fake T),
        D_T : Discriminator T(discriminates real and fake T)
        E : Encoder (Encodes real/fake T to feature space)
        C : Segmentor
        U : Decoder or Generator S (Generates fake S, recreates S)
        D_S : Discriminator S(discriminates real and fake S, also used for auxiliary feature space)
        D_P : Discriminator Mask(discriminates mask of S and T)
        """

        # self.netE =

        self.netC = networks.define_C(opt.seg_input_nc, opt.seg_output_nc, opt.netC, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain: # define generator_T, decoder and discriminators
            self.netG_T = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG_T, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netU = networks.define_G(opt.decoder_input_nc, opt.decoder_output_nc, opt.netDecoder, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netD_T = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netD_S = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netD_P = networks.define_D(opt.output_mask_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            self.fake_S_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            self.fake_T_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            #define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle - torch.nn.L1Loss()
            
