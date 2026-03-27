import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
# class mse_loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.loss_fn = nn.MSELoss()
#     def forward(self, output, target):
#         return self.loss_fn(output, target)

def ssim_loss(img1,img2,c1=1e-4,c2=9e-4):
    mu1=F.avg_pool2d(img1,3,1,1)
    mu2=F.avg_pool2d(img2,3,1,1)
    sigma1=F.avg_pool2d(img1*img1,3,1,1)-mu1*mu1
    sigma2=F.avg_pool2d(img2*img2,3,1,1)-mu2*mu2
    sigma12=F.avg_pool2d(img1*img2,3,1,1)-mu1*mu2
    ssim_n=(2*mu1*mu2+c1)*(2*sigma12+c2)
    ssim_d=(mu1*mu1+mu2*mu2+c1)*(sigma1+sigma2+c2)
    ssim_map=ssim_n/ssim_d
    return 1-ssim_map.mean()

def mse_loss(output, target):
    '''
    loss_r = F.mse_loss(output[:, 0, :, :], target[:, 0, :, :])  # R通道
    loss_g = F.mse_loss(output[:, 1, :, :], target[:, 1, :, :])  # G通道  
    loss_b = F.mse_loss(output[:, 2, :, :], target[:, 2, :, :])  # B通道
    '''

    return F.mse_loss(output, target)

def multi_loss(output,target):

    return F.mse_loss(output, target)+0.01*ssim_loss(output,target)


def l1_loss(output, target):
    return F.l1_loss(output, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

