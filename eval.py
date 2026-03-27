from cleanfid import fid
from core.base_dataset import BaseDataset
from models.metric import inception_score
from tqdm import tqdm
import numpy as np
from glob import glob
import os
from skimage.metrics import peak_signal_noise_ratio as psnr_cal
from skimage.metrics import structural_similarity as ssim_cal
from PIL import Image
import lpips
import torch
import subprocess
import sys
import shutil
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True, 
                    help='Path to the exp folder')

args = parser.parse_args()

exp_path = args.path

def calculate_fid(src, dst):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if sys.platform == "win32":
        num_workers = 0
    else:
        num_workers = 8
    subprocess.run(
        ["python", "-m", "pytorch_fid",  f"{src}", f"{dst}", "--device", f"{device}", "--batch-size", "8", "--num-workers", f"{num_workers}"]
    )


def calculate_lpips(src, dst):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(device)
    lpips_list = []
    for i, j in tqdm(zip(glob(src + "/*"), glob(dst + "/*")), total=len(glob(src + "/*"))):
        img1 = lpips.im2tensor(lpips.load_image(i)).to(device)
        img2 = lpips.im2tensor(lpips.load_image(j)).to(device)
        lpips_list.append(loss_fn_alex(img1, img2).item())
    print(f"LPIPS:{np.mean(lpips_list)}")

if __name__ == '__main__':

    ''' parser configs '''
    #/home/jovyan/code/DiffCR/experiments/test_naf_mul_swin_maskex_correct_interval_200_278_3_2_correct_interval_200_278_3_260121_153008
    #src = "experiments/test_naf_mul_swin_maskex_correct_interval_200_278_3/results/test/0/GT*"
    #dst = "experiments/test_naf_mul_swin_maskex_correct_interval_200_278_3/results/test/0/Out*"

    src=os.path.join(exp_path,'results/test/0/GT*')
    dst=os.path.join(exp_path,'results/test/0/Out*')
    psnr = []
    ssim = []
    for gt_path, out_path in tqdm(zip(sorted(glob(src)), sorted(glob(dst))), total=len(glob(src))):
        gt = np.array(Image.open(gt_path))
        out = np.array(Image.open(out_path))
        _psnr = psnr_cal(gt, out, data_range=255)
        _ssim = ssim_cal(gt, out, data_range=255, channel_axis=2)
        psnr += [_psnr]
        ssim += [_ssim]
    psnr = sum(psnr)/len(psnr)
    ssim = sum(ssim)/len(ssim)
    print(
        f'PSNR: {psnr}\n',
        f'SSIM: {ssim}\n',
    )

    #base_dir='experiments/test_naf_mul_swin_maskex_correct_interval_200_278_3/results/test/0' 
    #temp_dir='experiments/test_naf_mul_swin_maskex_correct_interval_200_278_3/temp'
    base_dir=os.path.join(exp_path,'results/test/0')
    temp_dir=os.path.join(exp_path,'temp')
    if not os.path.exists(temp_dir):
        gt_dir = os.path.join(temp_dir, "GT")
        out_dir = os.path.join(temp_dir, "Out")
        os.makedirs(gt_dir)
        os.makedirs(out_dir)

        # 复制GT开头的图片到GT目录
        for filename in os.listdir(base_dir):
            if filename.startswith('GT') and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(base_dir, filename)
                dst_path = os.path.join(gt_dir, filename[3:])
                shutil.copy2(src_path, dst_path)
        
        # 复制OUT开头的图片到OUT目录
        for filename in os.listdir(base_dir):
            if filename.startswith('Out') and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(base_dir, filename)
                dst_path = os.path.join(out_dir, filename[4:])
                shutil.copy2(src_path, dst_path)
    
    src2 = os.path.join(temp_dir, "GT")
    dst2 = os.path.join(temp_dir, "Out")
    calculate_fid(src2, dst2)
    calculate_lpips(src2, dst2)
    
    '''
    fid_score = fid.compute_fid(os.path.join(temp_dir,'GT'),os.path.join(temp_dir,'Out'))
    is_mean, is_std = inception_score(
        BaseDataset(glob(dst)),
        cuda=True,
        batch_size=8,
        resize=True,
        splits=10,
    )
    print(
        f'FID: {fid_score}\n',
        f'IS: {is_mean} {is_std}\n',
    )
    '''
    
