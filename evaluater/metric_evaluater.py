# SSIM, PSNR, Pearson CC.
import mrcfile
import glob
import os
import numpy as np
import math

rawBack_DIR      = "/cdata/thesis/rawBack/"
rawSign_DIR      = "/cdata/thesis/rawSign/"
raw_DIR          = "/cdata/thesis/norm/bench10077_27/"
# NT2CdenoisedBack_DIR = "/cdata/thesis/pipeTemp500Back/"
# NT2CdenoisedSign_DIR = "/cdata/thesis/pipeTemp500Sign/"
# NT2Cdenoised_DIR     = "/cdata/thesis/pipeTemp_500_fineDenoise_27/"
# TOPAZdenoisedBack_DIR = "/cdata/thesis/topazGenBack/"
# TOPAZdenoisedSign_DIR = "/cdata/thesis/topazGenSign/"
# TOPAZdenoised_DIR     = "/cdata/thesis/topazGeneral/"
NT2CdenoisedBack_DIR = "/cdata/thesis/norm/pipeTemp500Back/"
NT2CdenoisedSign_DIR = "/cdata/thesis/norm/pipeTemp500Sign/"
NT2Cdenoised_DIR     = "/cdata/thesis/norm/pipeTemp500Whole/"
TOPAZdenoisedBack_DIR = "/cdata/thesis/norm/topazGenBack/"
TOPAZdenoisedSign_DIR = "/cdata/thesis/norm/topazGenSign/"
TOPAZdenoised_DIR     = "/cdata/thesis/norm/topazGeneral/"

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(np.max(img1) / math.sqrt(mse))

# def cal_raw_max(A):
#     max = 0.0
#     for a in A:
#         a_data = mrcfile.open(a, permissive=True).data.copy()
#         max = np.max(a_data) if np.max(a_data) > max else max
#     return max

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

print("NT2C")
# calculate_metrics(rawBack_DIR, NT2CdenoisedBack_DIR, "Background")
# calculate_metrics(rawSign_DIR, NT2CdenoisedSign_DIR, "Signal")
calculate_metrics(raw_DIR, NT2Cdenoised_DIR, "Whole")
print("TOPAZ")
# calculate_metrics(rawBack_DIR, TOPAZdenoisedBack_DIR, "Background")
# calculate_metrics(rawSign_DIR, TOPAZdenoisedSign_DIR, "Signal")
calculate_metrics(raw_DIR, TOPAZdenoised_DIR, "Whole")