import math
import os
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from data.cloud import CloudOcclusion
from .ours.maskex import Mask_Ex
import pdb
import cv2
import random
import time
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def saveimg(img,name):
    if img.min()<0:
        img=torch.clip((img/2+0.5),0.0,1.0).squeeze()
    else:
        img=torch.clip(img,0.0,1.0).squeeze()
    img=(img*255.0).cpu().numpy()
    if len(img.shape)==3:
        img=img.transpose(1,2,0)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    os.makedirs('temp_exp',exist_ok=True)
    cv2.imwrite(os.path.join('temp_exp',name),img)

class Network(BaseNetwork):
    def __init__(self, unet, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == "swin":
            from .ours.swin import UNet
        elif module_name == "naf_mul_swin":
            from .ours.naf_mul_swin import UNet
        self.denoise_fn = UNet(**unet)

        self.maskex_dir = "pretrained/maskex.pth"
        self.maskex_model=Mask_Ex().cuda()
        self.maskex_model.load_state_dict(torch.load(self.maskex_dir))
        self.maskex_model.eval()
        self.quantize_interval=10
        
        self.compare_list=[60, 101, 147, 194, 237, 281, 326, 363, 400, 401, 402, 411, 429, 446, 460, 486, 533, 
        582, 651, 739, 822, 925, 1052, 1200, 1357, 1534, 1717, 1901, 2089, 2280, 2495, 2728, 2997, 3258, 3530, 
        3810, 4099, 4403, 4714, 5041, 5374, 5720, 6064, 6413, 6792, 7195, 7622, 8078, 8570, 9093, 9606, 10140, 
        10692, 11227, 11760, 12311, 12841, 13366, 13870, 14385, 14918, 15455, 16030, 16610, 17201, 17811, 18430, 
        19031, 19619, 20189, 20758, 21309, 21877, 22428, 22966, 23521, 24075, 24672, 25295, 25905, 26520, 27145, 
        27768, 28393, 29004, 29621, 30237, 30852, 31447, 32044, 32646, 33222, 33799, 34378, 34932, 35508, 36069, 
        36621, 37175, 37718, 38259, 38787, 39303, 39811, 40311, 40825, 41315, 41796, 42261, 42734, 43202, 43675, 
        44139, 44596, 45045, 45484, 45904, 46313, 46723, 47122, 47518, 47912, 48303, 48673, 49035, 49391, 49743, 
        50083, 50420, 50744, 51073, 51401, 51713, 52028, 52335, 52637, 52937, 53238, 53534, 53817, 54076, 54338, 
        54600, 54844, 55078, 55311, 55533, 55740, 55946, 56156, 56358, 56557, 56745, 56922, 57101, 57270, 57438, 
        57599, 57753, 57900, 58047, 58192, 58333, 58468, 58609, 58743, 58874, 59001, 59134, 59256, 59381, 59502, 
        59630, 59750, 59868, 59989, 60103, 60214, 60326, 60433, 60536, 60640, 60737, 60836, 60933, 61028, 61123, 
        61217, 61308, 61399, 61483, 61564, 61649, 61730, 61806, 61883, 61961, 62032, 62101, 62171]
        

        seed = (os.getpid() ^ int(time.time() * 1e9)) & 0xFFFFFFFF
        random.seed(seed)
        

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        thickness=np.linspace(0,0.9,2000)
        thickness=(thickness**0.2)/2.0
        self.thickness=thickness
        


    @torch.no_grad()
    #def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
    def restoration(self, y_cond, y_0=None, sample_num=8):
        perlin_flag=1
        res_epoch=100 #step
        res_epoch-=1
        b, *_ = y_cond.shape #batch_size
        y_t1 = y_cond[:,:3,:,:]
        y_t2 = y_cond[:,3:6,:,:]
        y_t3 = y_cond[:,6:9,:,:]

        count1=(self.maskex_model(y_t1)>0).sum().item()
        count2=(self.maskex_model(y_t2)>0).sum().item()
        count3=(self.maskex_model(y_t3)>0).sum().item()

        count = min(count1,count2,count3)
        if count==count1:
            y_t=y_t1
            flag=1
        elif count==count2:
            y_t=y_t2
            flag=2
        else:
            y_t=y_t3
            flag=3
        ret_arr = y_t
        #cv2.imwrite('y_t.png',y_t)
        closest_index = min(range(len(self.compare_list)), key=lambda i: abs(self.compare_list[i] - count))
        t = closest_index*self.quantize_interval
        t = torch.full((b,), t, device=y_cond.device, dtype=torch.long) 
        y_0_hat = self.denoise_fn(torch.cat([y_cond, y_t], dim=1).float(), t)
        
        y_cloud=1.0*torch.ones(1,3,256,256).cuda()


        while res_epoch>=1:
            if t[0]<=50:
                res_epoch=0
            else:           
                if perlin_flag==1:
                    t=int(t[0])
                    y_0_hat[y_0_hat>=0.992]=0.992 #avoid /0
                    M_t=((y_t-y_0_hat)/(y_cloud-y_0_hat)).squeeze().mean(dim=0) #256*256
                    #saveimg(M_t,'m1.png')
                    thres_t=1-1.2*self.thickness[t]
                    width_t=0.1*(1-0.8*self.thickness[t])
                    perlin_map=(M_t-1/2)*2*width_t + thres_t  #C(x,y),256*256
                    #saveimg(perlin_map,'perlin.png')
                    perlin_flag=0

                t=int(t)-self.quantize_interval #k->k-kmin
                thres_t=1-1.2*self.thickness[t]
                width_t=0.05*(1-0.8*self.thickness[t])
                cmask=torch.clip((perlin_map-thres_t+width_t)/(2*width_t),0.0,1.0).unsqueeze(0).unsqueeze(0)
                cmask=cmask*cmask*(3-2*cmask)
                y_t=cmask*y_cloud + (1-cmask)*y_0_hat
                #saveimg(y_t,f'addnoise_{t}.png')
                t = torch.full((b,), t, device=y_cond.device, dtype=torch.long)
                y_0_hat = self.denoise_fn(torch.cat([y_cond, y_t], dim=1).float(), t)
                res_epoch-=1

        return y_0_hat, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None,flag=0):
        if flag==0:  #y1 as input
            y_noisy = y_cond[:,:3,:,:]
            sample_gammas=torch.zeros([len(y_noisy)]).cuda()
            for i in range(len(y_noisy)):
                with torch.no_grad():
                    self.maskex_model.eval()
                    count =(self.maskex_model(y_noisy[i].unsqueeze(0))>0).sum().item()

                closest_index = min(range(len(self.compare_list)), key=lambda i: abs(self.compare_list[i] - count))
                sample_gammas[i]=closest_index*self.quantize_interval
        else:  #y1+perlin as input
            y_noisy = y_cond[:,:3,:,:]  #y1
            cloud_gen = CloudOcclusion(y_noisy)
            y_noisy, sample_gammas = cloud_gen.apply_cloud_occlusion(self.thickness,0.05)#change edge_blur here
            for i in range(len(y_noisy)): 
                with torch.no_grad():
                    self.maskex_model.eval()
                    count =(self.maskex_model(y_noisy[i].unsqueeze(0))>0).sum().item()

                closest_index = min(range(len(self.compare_list)), key=lambda i: abs(self.compare_list[i] - count))
                sample_gammas[i]=closest_index*self.quantize_interval

        

        if mask is not None:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)
            loss = self.loss_fn(mask*noise, mask*noise_hat)
        else:
            y_0_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            loss = self.loss_fn(y_0, y_0_hat)
        return loss




