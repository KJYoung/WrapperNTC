# SSIM, PSNR, Pearson CC.
import mrcfile
import glob
import os
import numpy as np
import math

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(np.max(img1) / math.sqrt(mse))

def calculate_metrics(raw_DIR, denoised_DIR, msg=""):
    # prepare the path lists.
    A = []
    B = []
    for path in glob.glob(raw_DIR + '*.mrc'):
        name = os.path.basename(path)
        A.append(path)
        B.append(denoised_DIR + name)
    
    # raw_max = cal_raw_max(A)
    psnr = []
    for a, b in zip(A, B):
        a_data = mrcfile.open(a, permissive=True).data.copy()
        b_data = mrcfile.open(b, permissive=True).data.copy()
        psnr.append(calculate_psnr(a_data, b_data))
        # print(f"PSNR : {calculate_psnr(a_data, b_data, raw_max)}")
    print(f"{msg} PSNR average : {np.mean(psnr)}")