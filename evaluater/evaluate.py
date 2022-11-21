import os
import sys
import glob
from math import floor, ceil
import mrcfile
from TARGET import back_patches, signal_patches
from metric_evaluater import calculate_metrics
import numpy as np
# USAGE ################################################################################
# python evaluate.py [model_path] [output_dir] [skip_to_go_metric]
# ex. python /cdata/NT2C/evaluater/evaluate.py /cdata/pipeTemp/fineModel/model_epoch500.sav /cdata/thesis/pipeTemp_500_fineDenoise_27/ 
# ex. python /cdata/NT2C/evaluater/evaluate.py skip /cdata/thesis/topazGeneral/  (/cdata/thesis/topazGeneral/denoised/ should be filled with topaz denoised.)
########################################################################################

#input
model_path        = sys.argv[1]
output_dir        = sys.argv[2]
skip_to_go_metric = sys.argv[3]

cuda = 6
NT2C_DIR = "/cdata/NT2C/"

raw_wholeNorm = "/cdata/thesis/raw/wholeNorm/"
raw_backNorm  = "/cdata/thesis/raw/backNorm/"
raw_signNorm  = "/cdata/thesis/raw/signNorm/"

RAW_DATA_PATH = "/cdata/thesis/raw/whole/*.mrc"

# 0. Prepare Dirs.
out_root      = output_dir
out_denoised  = output_dir + "denoised/"
out_wholeNorm = output_dir + "wholeNorm/"
out_backNorm  = output_dir + "backNorm/"
out_signNorm  = output_dir + "signNorm/"

if not os.path.exists(out_root):
    os.makedirs(out_root)
if not os.path.exists(out_denoised):
    os.makedirs(out_denoised)
if not os.path.exists(out_wholeNorm):
    os.makedirs(out_wholeNorm)
if not os.path.exists(out_backNorm):
    os.makedirs(out_backNorm)
if not os.path.exists(out_signNorm):
    os.makedirs(out_signNorm)

input_DIR = out_denoised
outputBACK_DIR = out_backNorm
outputSIGN_DIR = out_signNorm
outputWHOLE_DIR = out_wholeNorm

if skip_to_go_metric == 'false':
    # 1. Fine Denoise.
    if model_path != 'skip':
        status1 = os.system(f'python {NT2C_DIR}denoiser/denoise_cmd.py -o {out_denoised} -m {model_path} -c 512 -d {cuda} {RAW_DATA_PATH}')
        if status1 != 0:
            print("----Step 1 was not successfully finished with error code {}".format(status2))
            quit()
        print("---------------------------- Fine Denoise Finished. ")

    # 2. Extract(with Normalize)

    for file in back_patches:
        orig = mrcfile.open(f"{input_DIR}{file['file']}", permissive=True).data
        for i, patch in enumerate(file['patches']):
            orig_ = orig.copy()
            orig_ = orig_[floor(patch[2]):ceil(patch[3]), floor(patch[0]):ceil(patch[1])]
            with mrcfile.new(f"{outputBACK_DIR}{file['file']}_{i}.mrc") as mrc:
                mrc.set_data(orig_ / np.max(orig_))

    for file in signal_patches:
        orig = mrcfile.open(f"{input_DIR}{file['file']}", permissive=True).data
        for i, patch in enumerate(file['patches']):
            orig_ = orig.copy()
            orig_ = orig_[floor(patch[2]):ceil(patch[3]), floor(patch[0]):ceil(patch[1])]
            with mrcfile.new(f"{outputSIGN_DIR}{file['file']}_{i}.mrc") as mrc:
                mrc.set_data(orig_ / np.max(orig_))

    A = []
    B = []
    for path in glob.glob(input_DIR + '*.mrc'):
        name = os.path.basename(path)
        A.append(path)
        B.append(outputWHOLE_DIR + name)

    for ins, outs in zip(A, B):
        orig = mrcfile.open(f"{ins}", permissive=True).data.copy()
        with mrcfile.new(f"{outs}") as mrc:
            mrc.set_data(orig / np.max(orig))
    print("---------------------------- Patch Extract Finished.")

# 3. metric evaluater

calculate_metrics(raw_backNorm, outputBACK_DIR, "background")
calculate_metrics(raw_signNorm, outputSIGN_DIR, "signal")
calculate_metrics(raw_wholeNorm, outputWHOLE_DIR, "whole")

print("---------------------------- Metric evaluation Finished. ")