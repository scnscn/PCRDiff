import numpy as np
import math
from PIL import Image
import noise
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import random
import torch
import pdb
import os
import time
class CloudOcclusion:
    def __init__(self, image, cmask=None,cloud_color=np.array([255/255,255/255,255/255]), shadow_color=np.array([-225/255,-225/255,-205/255]), random_seed=None, random_seed1=None, shadow_softness=1.0): #对应云层颜色（240，240，245），image为[8,3,256,256]
        
        
        seed = (os.getpid() ^ int(time.time() * 1e9)) & 0xFFFFFFFF
        random.seed(seed)
        self.original_img = image #8,3,256,256,torch,[-1,1]
        self.batch, _, self.height, self.width= self.original_img.shape
        self.cloud_color = torch.from_numpy(cloud_color).view(3,1,1).cuda() #torch
        self.shadow_color = torch.from_numpy(shadow_color).view(3,1,1).cuda()
        self.shadow_softness=shadow_softness 
        self.cmask=cmask
        
        
        if random_seed is None:
            random_seed = random.randint(-3000, 3000)
        self.random_seed = random_seed
        if random_seed1 is None:
            random_seed1 = random.randint(0,1)
        self.random_seed1 = random_seed1
        
        
        self.base_cloud = self._generate_base_cloud() 
    
    def _generate_base_cloud(self, scale=0.005, octaves=8, persistence=0.6, lacunarity=2.0):
        """
        generate base cloud
        
        Returns:
            torch[0,1]
        """
        cloud_texture = np.zeros((2*self.height, 2*self.width))
        
        for y in range(2*self.height):
            for x in range(2*self.height):
                value = noise.pnoise2(x * scale, 
                                    y * scale,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=1024,
                                    repeaty=1024,
                                    base=self.random_seed
                                    )
                # [-1,1]->[0,1]
                cloud_texture[y, x] = (value + 1) / 2
        
        return torch.from_numpy(cloud_texture).cuda() #torch,[-1,1],512*512
    
    def smoothstep(self, edge0, edge1, x):
        x = torch.clip((x - edge0) / (edge1 - edge0), min=0.0, max=1.0)
        return x * x * (3 - 2 * x) #torch
    
    def generate_cloud_mask(self, thickness, edge_blur=0.1): #thickness -> np
        """
        Returns:
            cloudmask[0,1]
        """
        threshold = torch.tensor(1 - thickness * 1.2).cuda()
        blur_width = torch.tensor(edge_blur * (1 - thickness * 0.8)).cuda()
        
        mask_cloud = self.smoothstep(threshold - blur_width, 
                             threshold + blur_width, 
                             self.base_cloud) #torch
        
        return torch.clip(mask_cloud, min=0, max=1)
    
    def _apply_gaussian_blur(self, tensor, sigma):
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(tensor.detach().cpu().numpy(), sigma=sigma)
        return torch.from_numpy(blurred).cuda().float()
    
    def apply_cloud_occlusion(self, thickness, edge_blur=0.1, cloud_flag=None):
        result = self.original_img
        random.seed(42)
        for i in range(len(self.original_img)):
            if not cloud_flag:
                cloud_flag = random.randint(0,500)
            if i==0:
                sample_thickness=torch.tensor([cloud_flag])
            else:
                sample_thickness=torch.concat([sample_thickness,torch.tensor([cloud_flag])])
            cloud_mask = self.generate_cloud_mask(thickness[cloud_flag], edge_blur)
    
            if self.cmask is None:
                cloud_mask_rgb = torch.stack([cloud_mask] * 3, dim=0)[:,128:384,128:384]
                #shadow_mask_rgb = torch.stack([shadow_mask] * 3, dim=0)[:,128:384,128:384]
            else:
                cloud_mask_rgb = torch.stack([cloud_mask] * 3, dim=0)[:,128:384,128:384]*self.cmask
                #shadow_mask_rgb = torch.stack([shadow_mask] * 3, dim=0)[:,128:384,128:384]*self.cmask
                

            result[i] = (1 - cloud_mask_rgb) * result[i] + cloud_mask_rgb * self.cloud_color
            #result[i] = (1 - cloud_mask_rgb-shadow_mask_rgb) * result[i] + cloud_mask_rgb * self.cloud_color + shadow_mask_rgb * self.shadow_color
        result = torch.clip(result, -1., 1.).cuda()
        
        sample_thickness=sample_thickness.unsqueeze(-1).cuda()
        return result,sample_thickness

def image_read_rgb(image_path):
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        img = (img / 1.0).transpose((2, 0, 1))

        image = torch.from_numpy((img.copy())).float()
        
        image = image / 255.0
        mean = torch.as_tensor([0.5, 0.5, 0.5],
                               dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5],
                              dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std) #[0-255]->[-1,1]

        return image
    
