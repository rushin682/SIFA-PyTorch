import torch
import torch.nn.functional as F
import numpy as np

def pixel_wise_softmax(output_map):
    return torch.nn.Softmax2d(output_map, dim=1)

def jaccard(conf_matrix):
    num_cls = conf_matrix.shape[0]
    jac = np.zeros(num_cls)
    for ii in range(num_cls):
        pp = np.sum(conf_matrix[:, ii])
        gp = np.sum(conf_matrix[ii, :])
        hit = conf_matrix[ii, ii]
        jac[ii] = hit * 1.0 / (pp + gp - hit)
    return jac

def dice(conf_matrix):

    num_cls = conf_matrix.shape[0]
    dic = np.zeros(num_cls)
    for ii in range(num_cls):
        pp = np.sum(conf_matrix[:,ii])
        gp = np.sum(conf_matrix[ii,:])
        hit = conf_matrix[ii,ii]
        if (pp + gp) == 0:
            dic[ii] = 0
        else:
            dic[ii] = 2.0 * hit / (pp + gp)
    return dic

def dice_eval(compact_pred, labels, n_class):

    dice_arr = []
    dice = 0
    eps = 1e-7
    pred = F.one_hot(compact_pred, depth = n_class)
    for i in xrange(n_class):
        inse = tf.reduce_sum(pred[:, :, :, i] * labels[:, :, :, i])
        union = tf.reduce_sum(pred[:, :, :, i]) + tf.reduce_sum(labels[:, :, :, i])
        dice = dice + 2.0 * inse / (union + eps)
        dice_arr.append(2.0 * inse / (union + eps))

    return dice_arr
