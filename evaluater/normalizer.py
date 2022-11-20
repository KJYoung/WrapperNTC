# SSIM, PSNR, Pearson CC.
import mrcfile
import glob
import os
import numpy as np
import math

# Back_DIR   = "/cdata/thesis/pipeTemp500Back/"
# Sign_DIR   = "/cdata/thesis/pipeTemp500Sign/"
# Whole_DIR  = "/cdata/thesis/pipeTemp_500_fineDenoise_27/"

# Back_OUT   = "/cdata/thesis/norm/pipeTemp500Back/"
# Sign_OUT   = "/cdata/thesis/norm/pipeTemp500Sign/"
# Whole_OUT  = "/cdata/thesis/norm/pipeTemp500Whole/"

Back_DIR   = "/cdata/thesis/topazGenBack/"
Sign_DIR   = "/cdata/thesis/topazGenSign/"
Whole_DIR  = "/cdata/thesis/topazGeneral/"

Back_OUT   = "/cdata/thesis/norm/topazGenBack/"
Sign_OUT   = "/cdata/thesis/norm/topazGenSign/"
Whole_OUT  = "/cdata/thesis/norm/topazGeneral/"


def normalize(raw_DIR, norm_DIR, msg=""):
    # prepare the path lists.
    A = []
    out = []
    for path in glob.glob(raw_DIR + '*.mrc'):
        name = os.path.basename(path)
        A.append(path)
        out.append(norm_DIR + name)
    
    # raw_max = cal_raw_max(A)
    psnr = []
    for file, output in zip(A, out):
        a_data = mrcfile.open(file, permissive=True).data.copy()
        norm_a = a_data / np.max(a_data)
        with mrcfile.new(output, overwrite=True) as mrc:
            mrc.set_data(norm_a)

# normalize("/cdata/benchDIR/bench10077_27/", "/cdata/thesis/norm/bench10077_27/")
normalize(Back_DIR, Sign_OUT)
normalize(Sign_DIR, Sign_OUT)
normalize(Whole_DIR, Whole_OUT)