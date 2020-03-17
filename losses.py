import torch
import torch.nn.functional as F


def cycle_consistency_loss(real_images, generated_images):
    """
    Compute the cycle consistency loss. or L1 Loss
    """
    cyclic_loss = torch.nn.L1Loss(reduction='mean')
    return cyclic_loss(real_images, generated_images)


def generator_loss(prob_fake_is_real):
    """
    Computes the LS-GAN loss as minimized by the generator. or MSE Loss
    """
    gan_loss = torch.nn.MSELoss(reduction='mean')
    return gan_loss(1, prob_fake_is_real)


def discriminator_loss(prob_real_is_real, prob_fake_is_real):
    """
    Computes the LS-GAN loss as minimized by the discriminator. or MSE Loss
    """
    gan_loss = torch.nn.MSELoss(reduction='mean')

    return gan_loss(1, prob_real_is_real) + gan_loss(0, prob_fake_is_real)


def _softmax_weighted_loss(logits, gt, num_classes=5):
    """
    Calculate weighted cross-entropy loss.
    """
    # compute softmax over the classes axis
    # input_soft = F.softmax2d(input, dim=1)
    # predi = input_soft[:,i,:,:]

    weigths = []
    for i in range(num_classes):
        gti = gt[:,i,:,:]
        weighted = 1-(torch.sum(gti) / torch.sum(gt))
        weights.append(weighted)

    CELoss = torch.nn.CrossEntropyLoss(torch.FloatTensor(weights))

        # if i == 0:
        #     raw_loss = -1.0 * weighted * gti * torch.log(torch.clamp(predi, 0.005, 1))
        # else:
        #     raw_loss += -1.0 * weighted * gti * torch.log(torch.clamp(predi, 0.005, 1))

    loss = CELoss(logits, gt)

    return loss


def _dice_loss(logits, gt, num_classes=5):
    """
    Calculate dice loss.
    """
    dice = 0
    eps = 1e-7

    # compute softmax over the classes axis
    input_soft = F.softmax2d(input, dim=1)

    # The labels are already one hot tensors

    # Computing the Actual Dice Score
    for i in range(num_classes):
        inse = torch.sum(input_soft[:, i, :, :]*gt[:, i, :, :])
        l = torch.sum(input_soft[:, i, :, :]*input_soft[:, i, :, :])
        r = torch.sum(gt[:, i, :, :])
        dice += 2.0 * inse/(l+r+eps)

    return 1 - 1.0 * dice / 5


def task_loss(prediction, gt, num_classes=5):
    """
    Calculate task loss, which consists of the weighted cross entropy loss and dice loss
    """
    ce_loss = _softmax_weighted_loss(prediction, gt, num_classes)
    dice_loss = _dice_loss(prediction, gt, num_classes)

    return ce_loss, dice_loss
