import torch
import torch.nn.functional as F
# from torch.autograd import Variable
# import numpy as np


def gray_mask(cam):
    n, c, h, w = cam.size()
    with torch.no_grad():
        cam_d = F.relu(cam.detach())
        cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1) + 1e-5
        cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
        cam_d_norm[:, 0, :, :] = 1 - torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0]
        cam_max = torch.max(cam_d_norm[:, 1:, :, :], dim=1, keepdim=True)[0]
        cam_d_norm[:, 1:, :, :][cam_d_norm[:, 1:, :, :] < cam_max] = 0
        cam_d_norm[:, 1:, :, :][cam_d_norm[:, 1:, :, :] != 0] = 1
    dilation_filter = torch.ones((7, 7))
    dilation_filter = torch.tensor(dilation_filter, dtype=torch.float32).unsqueeze(0).expand(n*c, 1, 7, 7).cuda()
    smooth_mask = F.conv2d(cam_d_norm.view(1, n*c, h, w), dilation_filter, stride=1, padding=3, groups=n*c).view(n, c, h, w)
    smooth_mask[smooth_mask != 0] = 1

    return cam_d_norm, smooth_mask


    # n,c,h,w = cam.shape
    # mask = cam.detach().view(n,c,1,h*w)
    # # mask = mask.detach()
    # mean = torch.mean(mask, dim=-1).unsqueeze(2).expand(n,c,1,h*w) * 2
    # mean[mask >= mean] = 1
    # mask[mask < mean] = 0
    # mask = mask.view(n,c,h,w)
    # return mask * label


def laplacian_edge(img):
    n, c, h, w = img.shape
    if c == 21:
        img = img.view(1, n*c, h, w)
        lap_filter = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        lap_filter = torch.tensor(lap_filter, dtype=torch.float32).unsqueeze(0).expand(n*c, 1, 3 ,3)
        lap_filter = lap_filter.cuda()
        lap_edge = F.conv2d(img, lap_filter, stride=1, padding=1, groups=n*c)
        lap_edge = lap_edge.view(n, c, h, w)
        return lap_edge
    else:
        laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filter = torch.reshape(laplacian_filter, [1, 1, 3, 3])
        filter = filter.cuda()
        lap_edge = F.conv2d(img, filter, stride=1, padding=1)
        return lap_edge

def gradient_x(img):
    n, c, h, w = img.shape
    if c == 21:
        img = img.view(1, n*c, h, w)
        filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        filter = torch.tensor(filter, dtype=torch.float32).unsqueeze(0).expand(n*c, 1, 3 ,3)
        filter = filter.cuda()
        gx = F.conv2d(img, filter, stride=1, padding=1, groups=n*c)
        gx = gx.view(n, c, h, w)
        return gx
    else:
        sobel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        filter = torch.reshape(sobel,[1,1,3,3])
        filter = filter.cuda()
        gx = F.conv2d(img, filter, stride=1, padding=1)
        return gx


def gradient_y(img):
    n, c, h, w = img.shape
    if c == 21:
        img = img.view(1, n*c, h, w)
        filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        filter = torch.tensor(filter, dtype=torch.float32).unsqueeze(0).expand(n*c, 1, 3 ,3)
        filter = filter.cuda()
        gy = F.conv2d(img, filter, stride=1, padding=1, groups=n*c)
        gy = gy.view(n, c, h, w)
        return gy
    else:
        sobel = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        filter = torch.reshape(sobel, [1, 1,3,3])
        filter = filter.cuda()
        gy = F.conv2d(img, filter, stride=1, padding=1)
        return gy

def charbonnier_penalty(s):
    cp_s = torch.pow(torch.pow(s, 2) + 0.001**2, 0.5)
    return cp_s

def get_saliency_smoothness(pred, gt, label, size_average=True):
    alpha = 10
    s0 = 10
    s1 = 10
    s2 = 1
    ## Obtain Gate
    # confidence_mask, smoothness_mask = gray_mask(pred)

    ## first oder derivative: sobel
    sal_x = torch.abs(gradient_x(pred))
    sal_y = torch.abs(gradient_y(pred))
    # gt_x = gradient_x(gt)
    # gt_y = gradient_y(gt)
    w_x = torch.exp(gt * (-alpha))
    w_y = torch.exp(gt * (-alpha))
    cps_x = charbonnier_penalty(sal_x * w_x)
    cps_y = charbonnier_penalty(sal_y * w_y)
    cps_xy = cps_x + cps_y

    ## second order derivative: laplacian
    lap_sal = torch.abs(laplacian_edge(pred))
    # lap_gt = torch.abs(laplacian_edge(gt))
    weight_lap = torch.exp(gt * (-alpha))
    weighted_lap = charbonnier_penalty(lap_sal*weight_lap)


    smooth_loss = s1*torch.mean(cps_xy * label) + s2*torch.mean(weighted_lap * label)

    return smooth_loss

class smoothness_loss(torch.nn.Module):
    def __init__(self, size_average = True):
        super(smoothness_loss, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target, label):

        return get_saliency_smoothness(pred, target, label, self.size_average)
