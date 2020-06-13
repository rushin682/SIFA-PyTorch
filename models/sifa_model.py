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

        self.netE = networks.define_E(opt.encoder_input_nc, opt.encoder_output_nc, opt.ngf, opt.netE, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netC = networks.define_C(opt.seg_input_nc, opt.seg_output_nc, opt.netC, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain: # define generator_T, decoder and discriminators
            self.netG_T = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG_T, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netU = networks.define_G(opt.decoder_input_nc, opt.decoder_output_nc, opt.netDecoder, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netD_T = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # Discriminator_S i.e with an auxiliary o/p
            self.netD_S = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netD_P = networks.define_D(opt.output_mask_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            self.fake_S_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            self.fake_T_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            #define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss().to(self.device)
            self.criterionSeg_Dice = networks.DICELoss(num_classes=5).to(self.device) # num_classes = opt.num_classes
            self.criterionSeg_CE = networks.WCELoss(num_classes=5).to(self.device) # num_classes = opt.num_classes
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_T = torch.optim.Adam(self.netG_T.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_U = torch.optim.Adam(self.netU.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_T = torch.optim.Adam(self.netD_T.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_S = torch.optim.Adam(self.netD_S.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_Seg = torch.optim.Adam(itertools.chain(self.netE.parameters(), self.netC.parameters()),
                                                  lr=opt.seg_lr, betas=(opt.beta1, 0.999), weight_decay=opt.decay_rate)


            self.optimizers.extend([self.optimizer_G_T, self.optimizer_U])
            self.optimizers.extend([self.optimizer_D_T, self.optimizer_D_S, self.optimizer_D_P])
            self.optimizers.extend([self.optimizer_Seg])

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.fake_T = self.netG_T(self.real_S)  # G_T(S)
        self.fake_latent_T = self.netE(self.fake_T) # E(fake_T)
        self.rec_S = self.netU(self.fake_latent_T)   # D(E(fake_T))
        self.mask_fake_T = self.netC(self.fake_latent_T) # C(E(fake_T))


        self.latent_T = self.netE(self.real_T)  # E(T)
        self.fake_S = self.netU(self.latent_T) # D(E(T))
        self.rec_T = self.netG_T(self.fake_S)   # G_T(fake_S)
        self.mask_T = self.netC(self.latent_T) # C(E(T))

    def backward_G_T(self):
        """Calculate the loss for generators G_T"""
        lambda_S = self.opt.lambda_S
        lambda_T = self.opt.lambda_T

        # GAN Loss D_T(G_T(S))
        self.loss_G_T = self.criterionGAN(self.netD_T(self.fake_T), True)
        # Forward cycle loss || rec_S or U(E(G_T(S))) - S ||
        self.loss_cycle_S = self.criterionCycle(self.rec_S, self.real_S) * lambda_S
        # Backward cycle loss || rec_T or G_T(D(E(T))) - T ||
        self.loss_cycle_T = self.criterionCycle(self.rec_T, self.real_T) * lambda_T
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_T + self.loss_cycle_S + self.loss_cycle_T
        self.loss_G.backward()

    def backward_U(self):
        """Calculate the loss for generators U"""
        lambda_S = self.opt.lambda_S
        lambda_T = self.opt.lambda_T

        # GAN Loss D_S(U(S))
        self.loss_G_S = self.criterionGAN(self.netD_S(self.fake_S), True)
        # Forward cycle loss || rec_S or U(E(G_T(S))) - S ||
        self.loss_cycle_S = self.criterionCycle(self.rec_S, self.real_S) * lambda_S
        # Backward cycle loss || rec_T or G_T(D(E(T))) - T ||
        self.loss_cycle_T = self.criterionCycle(self.rec_T, self.real_T) * lambda_T
        # combined loss and calculate gradients
        self.loss_U = self.loss_G_S + self.loss_cycle_S + self.loss_cycle_T
        self.loss_U.backward()

    def backward_D_basic(self, netD, real, fake, use_aux=False):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
            use_aux (boolean)   -- Extra functionality for auxiliary feature space
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        if use_aux:
            pred_fake, pred_fake_aux = netD(fake.detach())
        else:
            pred_fake = netD(fake.detach())

        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        if use_aux:
            pred_rec = netD(self.rec_S.detach())

            loss_D_rec = self.criterionGAN(pred_rec, True)
            loss_D_fake_aux = self.criterionGAN(pred_fake_aux, False)

            loss_D += (loss_D_rec + loss_D_fake_aux) * 0.5

        loss_D.backward()
        return loss_D


    def backward_D_T(self):
        """Calculate GAN loss for discriminator D_T"""
        fake_T = self.fake_T_pool.query(self.fake_T)
        self.loss_D_T = self.backward_D_basic(self.netD_T, self.real_T, fake_T)

    def backward_D_S(self):
        """Calculate GAN loss for discriminator D_S"""
        fake_S = self.fake_S_pool.query(self.fake_S)
        self.loss_D_S = self.backward_D_basic(self.netD_S, self.real_S, fake_S, use_aux=True)

    def backward_D_P(self):
        """Calculate GAN loss for discriminator D_P"""
        self.loss_D_P = self.backward_D_basic(self.netD_P, self.mask_T, self.mask_fake_T)


    def backward_Seg(self):
        """Calculate total segmentation loss for encoder E and segmenter C"""

        # GAN Loss D_S(G_S(T))
        self.loss_G_S = self.criterionGAN(self.netD_S(self.fake_S)[0], True)
        # Forward cycle loss || rec_S or U(E(G_T(S))) - S ||
        self.loss_cycle_S = self.criterionCycle(self.rec_S, self.real_S) * lambda_S
        # Backward cycle loss || rec_T or G_T(D(E(T))) - T ||
        self.loss_cycle_T = self.criterionCycle(self.rec_T, self.real_T) * lambda_T

        # Adversarial loss via generated image space
        self.loss_aux_S = self.criterionGAN(self.netD_S(self.fake_S)[1], True)
        self.encoder_loss = self.loss_G_S + self.loss_cycle_S + self.loss_cycle_T + self.loss_aux_S

        # Adversarial loss via semantic prediction space
        self.loss_P = self.criterionGAN(self.netD_P(self.mask_T), True) # Should it be mask_T?
        self.dice_loss_T = self.criterionSeg_Dice(self.mask_fake_T, self.gt_S)
        self.ce_loss_T = self.criterionSeg_CE(self.mask_fake_T, self.gt_S)

        self.classifier_loss = self.ce_loss_T + self.dice_loss_T + (self.loss_p_weight_value * self.loss_P)

        # Segmentation Loss = dice loss + cross-entropy loss
        self.seg_loss = (0.1 * self.encoder_loss) + self.classifier_loss
        self.seg_loss.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_T
        self.set_requires_grad([self.netD_T], False)  # D_T require no gradients when optimizing G_T
        self.optimizer_G_T.zero_grad()                # set G_T's gradients to zero
        self.backward_G_T()                           # calculate gradients for G_T
        self.optimizer_G_T.step()                     # update G_T's weights

        # D_T
        self.set_requires_grad([self.netD_T], True) # D_T requires gradients to optimize itself.
        self.optimizer_D.zero_grad()                # set D_T's gradients to zero
        self.backward_D_T()                         # calculate gradients for D_T
        self.optimizer_D_T.step()                   # update D_T's weights

        # E + C
        self.set_requires_grad([self.netD_S, self.netD_P], False) # D_S and D_P require no gradients when optimizing E+C
        self.optimizer_Seg.zero_grad()                            # set (E+C)'s gradients to zero
        self.backward_Seg()                                       # calculate gradients for E+C
        self.optimizer_Seg.step()                                 # update {E,C}'s weights'

        # U
        self.optimizer_U.zero_grad() # set decoder U's gradients to zero
        self.backward_U()            # calculate gradients for U
        self.optimizer_U.step()      # update U's weights

        # D_S
        self.set_requires_grad([self.netD_S], True) # D_S requires gradients to optimize itself.
        self.optimizer_D_S.zero_grad()              # set D_S's gradients to zero
        self.backward_D_S()                         # calculating gradients for D_S
        self.optimizer_D_S.step()                   # update D_S's weights

        #D_P
        self.set_requires_grad([self.netD_P], True) #
        self.optimizer_D_P.zero_grad()
        self.backward_D_P()
        self.optimizer_D_P.step()
