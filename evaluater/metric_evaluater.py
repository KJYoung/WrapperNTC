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

def calculate_snr(back, sign, msg=""):
    back_files, sign_files = glob.glob(back + '*.mrc'), glob.glob(sign + '*.mrc')
    
    leng = min(len(back_files), len(sign_files))

    back_mu, back_std = [], []
    for i in range(leng):
        x = mrcfile.open(back_files[i], permissive=True).data
        back_mu.append(np.mean(x))
        back_std.append(np.std(x))

    sign_std = []
    for i in range(leng):
        s = mrcfile.open(sign_files[i], permissive=True).data.copy() - back_mu[i]
        sign_std.append(np.std(s))
    
    sums = 2 * np.sum(np.log10(sign_std) - np.log10(back_std))
    print(f"{msg} SNR : {sums * 10 / leng} (in dB)")

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)

def calculate_pearsoncc(clean, denoised, msg=""):
    # from scipy.stats.stats import pearsonr
    logger = open(f"{msg}.txt", 'w')

    A = []
    B = []
    for path in glob.glob(clean + '*.mrc'):
        name = os.path.basename(path)
        A.append(path)
        B.append(denoised + name)

    pncc = []
    for a, b in zip(A, B):
        aa = mrcfile.open(a, permissive=True).data.copy().reshape(4096 * 4096)
        bb = mrcfile.open(b, permissive=True).data.copy().reshape(4096 * 4096)
        
        res = pearson_def(aa, bb)
        pncc.append(res)
        logger.write(f"{res}")
        print(res)
    logger.write(f"Result average : {np.mean(pncc)}")
    print(np.mean(pncc))
    logger.close()