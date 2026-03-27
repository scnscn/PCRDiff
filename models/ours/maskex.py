import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import cv2
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pdb
import tifffile as tiff
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, hidden_channel, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channel, 1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, hidden_channel, rate1))
        modules.append(ASPPConv(in_channels, hidden_channel, rate2))
        modules.append(ASPPConv(in_channels, hidden_channel, rate3))
        modules.append(ASPPPooling(in_channels, hidden_channel))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * hidden_channel, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Mask_Ex(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck with ASPP
        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ASPP(in_channels=512, hidden_channel=256, atrous_rates=[4, 8, 16]),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (上采样路径) - 使用转置卷积或上采样+卷积
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 256 = 128(skip) + 128(up)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 128 = 64(skip) + 64(up)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 输出层 - 移除BatchNorm和激活函数
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self,x):
        e1 = self.enc1(x)    # (B, 64, H, W)
        p1 = self.pool1(e1)  # (B, 64, H/2, W/2)
        
        e2 = self.enc2(p1)   # (B, 128, H/2, W/2)
        p2 = self.pool2(e2)  # (B, 128, H/4, W/4)
        
        e3 = self.enc3(p2)   # (B, 256, H/4, W/4)
        p3 = self.pool3(e3)  # (B, 256, H/8, W/8)
        
        # Bottleneck
        b = self.middle(p3)  # (B, 256, H/8, W/8)
        
        # Decoder with skip connections
        d3 = self.up3(b)     # (B, 128, H/4, W/4)
        #pdb.set_trace()
        d3 = torch.cat([d3, p2], dim=1) 
        d3 = self.dec3(d3)   # (B, 128, H/4, W/4)
        
        d2 = self.up2(d3)    # (B, 64, H/2, W/2)
        d2 = torch.cat([d2, p1], dim=1)
        d2 = self.dec2(d2)   # (B, 64, H/2, W/2)
        
        d1 = self.up1(d2)    # (B, 32, H, W)
        d1 = self.dec1(d1)   # (B, 32, H, W)
        
        # 输出
        out = self.final(d1)  # (B, 1, H, W)
        return out


class SpacrsDataset(data.Dataset):
    def __init__(self,mode):
        self.traindir='D:/dataset/sparcsnew/train'
        self.valdir='D:/dataset/sparcsnew/val'
        self.mode=mode
        self.train_list_photo=os.listdir(os.path.join(self.traindir,'photo'))
        #self.train_list_cmask=os.listdir(os.path.join(self.traindir,'cmask'))
        self.val_list_photo=os.listdir(os.path.join(self.valdir,'photo'))
        #self.val_list_cmask=os.listdir(os.path.join(self.valdir,'cmask'))
        self.len_train=len(os.listdir(os.path.join(self.traindir,'photo')))
        self.len_val=len(os.listdir(os.path.join(self.valdir,'photo')))

    def __len__(self):
        if self.mode=='train':
            return self.len_train
        else:
            return self.len_val
        
    def __getitem__(self,idx):
        if self.mode=='train':
            img_name=self.train_list_photo[idx]
            img=cv2.imread(os.path.join(self.traindir,'photo',img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=(np.float32(img).transpose(2,0,1)/255.0-0.5)*2 #[0,255]->[0,1]->[-0.5,0.5]->[-1,1]
            img=(torch.from_numpy(img)).to(torch.float32)

            
            mask=cv2.imread(os.path.join(self.traindir,'cmask',img_name[:-9]+'cmask.png'),cv2.IMREAD_GRAYSCALE)
            mask=(torch.from_numpy(mask).unsqueeze(0)).to(torch.float32)/255.0
            '''
            mask = torch.where(mask == 0, 
                    torch.tensor(-1, dtype=mask.dtype, device=mask.device), 
                    mask)
            '''
            sample = {'image':img, 'mask':mask, 'name':img_name[:-10]} #chw,chw
        
        else:
            img_name=self.val_list_photo[idx]
            img=cv2.imread(os.path.join(self.valdir,'photo',img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=(np.float32(img).transpose(2,0,1)/255.0-0.5)*2
            img=(torch.from_numpy(img)).to(torch.float32)


            mask=cv2.imread(os.path.join(self.valdir,'cmask',img_name[:-9]+'cmask.png'),cv2.IMREAD_GRAYSCALE)
            mask=(torch.from_numpy(mask).unsqueeze(0)).to(torch.float32)/255.0

            sample = {'image':img, 'mask':mask, 'name':img_name[:-9]} #chw,chw

        return sample
    

class DiceBCELoss(nn.Module):
    """Dice Loss + BCE Loss"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(1,2,3))
        union = pred_sigmoid.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()
        
        return bce_loss + dice_loss

def image_read_pretiff(image_path): 
    img = tiff.imread(image_path)
    img = (img / 1.0).transpose((2, 0, 1))

    image = torch.from_numpy((img.copy())).float()
    r = image[0, :, :]
    g = image[1, :, :]
    b = image[2, :, :]

    r = torch.clip(r, 0, 2000)
    g = torch.clip(g, 0, 2000)
    b = torch.clip(b, 0, 2000)

    image = torch.stack((r, g, b))
    image = image / 2000.0
    mean = torch.as_tensor([0.5, 0.5, 0.5],
                            dtype=image.dtype, device=image.device)
    std = torch.as_tensor([0.5, 0.5, 0.5],
                            dtype=image.dtype, device=image.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    image.sub_(mean).div_(std) #[0-10000]->[0,2000]->[0,1]->[-1,1]
    return image

if __name__=='__main__':
    ckpt_dir="/home/jovyan/code/DiffCR/pretrained/maskex.pth"
    model=Mask_Ex().cuda()
    if ckpt_dir:
        model.load_state_dict(torch.load(ckpt_dir))

    epoch=100

    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
    model.eval()
    
    imgdir2='/home/jovyan/code/DiffCR/temp_exp/origin_cloudy.png'
    img2=cv2.imread(imgdir2)
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    img2=(np.float32(img2).transpose(2,0,1)/255.0-0.5)*2 #[0,255]->[0,1]->[-0.5,0.5]->[-1,1]
    img2=(torch.from_numpy(img2)).to(torch.float32).unsqueeze(0).cuda()
    pred3=model(img2)
    pred3=torch.sigmoid(pred3)
    pred3=torch.where(pred3>=0.5,1,0).squeeze()*255
    pred3=pred3.cpu().numpy()
    cv2.imwrite('temp_exp/pred3.png',pred3)
    r'''
    
    dataset1=SpacrsDataset(mode='train')
    dataloader1=data.DataLoader(dataset=dataset1,batch_size=8,shuffle=True,num_workers=4)
    dataset2=SpacrsDataset(mode='val')
    dataloader2=data.DataLoader(dataset=dataset2,batch_size=1,shuffle=False,num_workers=1)
    criterion=DiceBCELoss()

    for epoch_now in range(epoch):
        print(f'————————start train {epoch_now}————————')
        model.train()
        total_train_loss=0.0
        for i,batch in enumerate(tqdm(dataloader1)):
            optimizer.zero_grad()
            input=batch['image'].cuda()
            target=batch['mask'].cuda()
            pred=model(input)
            cv2.imwrite('pred.png',torch.where(torch.sigmoid(pred[0])>=0.5,1,0).squeeze().detach().cpu().numpy()*255)
            loss=criterion(pred,target)
            total_train_loss += loss
            loss.backward()
            optimizer.step()
        total_train_loss=total_train_loss/len(dataloader1)
        print(f'epoch:{epoch_now},loss:{total_train_loss}')
        
        if epoch_now%10==9:
            print('————————start val————————')
            model.eval()
            os.makedirs(f"D:/dataset/val_exp/pred_epoch_{epoch_now}",exist_ok=True)
            os.makedirs(f"D:/dataset/val_exp/ckpt_epoch_{epoch_now}",exist_ok=True)
            total_val_loss=0.0
            with torch.no_grad():
                for j,val_batch in enumerate(tqdm(dataloader2)):
                    input=val_batch['image'].cuda()
                    target=val_batch['mask'].cuda()
                    pred=model(input)
                    loss=criterion(pred,target)
                    total_val_loss += loss.item()
                    output_pred=torch.sigmoid(pred)
                    output_pred=torch.where(output_pred<0.5,torch.tensor(0,dtype=output_pred.dtype,device=output_pred.device),output_pred)
                    output_pred=torch.where(output_pred>=0.5,torch.tensor(1,dtype=output_pred.dtype,device=output_pred.device),output_pred)
                    output_pred=output_pred.squeeze()*255
                    output_pred=output_pred.cpu().numpy()     
                    cv2.imwrite(os.path.join(f'D:/dataset/val_exp/pred_epoch_{epoch_now}',val_batch['name'][0]+'cmask.png'),output_pred)
            avg_val_loss = total_val_loss / len(dataloader2)
            scheduler.step(avg_val_loss)
            print(f'————————val_loss:{avg_val_loss}————————')
            torch.save(model.state_dict(), os.path.join(f'D:/dataset/val_exp/ckpt_epoch_{epoch_now}','checkpoint_epoch.pth'))
            print('————————end val————————')
    print('————————end train————————')
    '''
    

