import torch

def cycle_consistency_loss(real_images, generated_images):
    """
    Compute the cycle consistency loss.
    """
    return torch.mean(torch.abs(real_images - generated_images))


def lsgan_loss_generator(prob_fake_is_real):
    """
    Computes the LS-GAN loss as minimized by the generator.
    """
    return torch.mean((1-prob_fake_is_real)**2)


def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
    """
    Computes the LS-GAN loss as minimized by the discriminator.
    """
    return (torch.mean((1-prob_real_is_real)**2) +
            torch.mean((prob_fake_is_real-0)**2)) * 0.5


def _softmax_weighted_loss(logits, gt, num_classes):
    """
    Calculate weighted cross-entropy loss.

    !!Should use built in functions for these instead!!
    """
    softmax = torch.nn.Softmax()
    softmaxpred = softmax(logits)
    '''questionanble'''

    for i in xrange(num_classes):
        gti = gt[:,i,:,:]
        predi = softmaxpred[:,i,:,:]
        weighted = 1-(torch.sum(gti) / torch.sum(gt))
        if i == 0:
            raw_loss = -1.0 * weighted * gti * torch.log(torch.clamp(predi, 0.005, 1))
        else:
            raw_loss += -1.0 * weighted * gti * torch.log(torch.clamp(predi, 0.005, 1))

    loss = torch.mean(raw_loss)

    return loss


def _dice_loss_fun(logits, gt, num_classes):
    """
    Calculate dice loss.
    """
    dice = 0
    eps = 1e-7

    softmax = torch.nn.Softmax()
    softmaxpred = softmax(logits)
    '''questionanble'''

    for i in xrange(num_classes):
        inse = torch.sum(softmaxpred[:, i, :, :]*gt[:, i, :, :])
        l = torch.sum(softmaxpred[:, i, :, :]*softmaxpred[:, i, :, :])
        r = torch.sum(gt[:, i, :, :])
        dice += 2.0 * inse/(l+r+eps)

    return 1 - 1.0 * dice / 5


def task_loss(prediction, g, num_classes):
    """
    Calculate task loss, which consists of the weighted cross entropy loss and dice loss
    """
    ce_loss = _softmax_weighted_loss(prediction, gt, num_classes)
    dice_loss = _dice_loss_fun(prediction, gt, num_classes)

    return ce_loss, dice_loss
