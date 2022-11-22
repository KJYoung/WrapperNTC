# Evaluate SIM dataset.
import os
import sys
import glob
from math import floor, ceil
import mrcfile
from metric_evaluater import calculate_metrics, calculate_pearsoncc
import numpy as np
from normalizer import normalize
# USAGE ################################################################################
# python evaluate.py [model_path] [output_dir] [skip_to_go_metric]
# ex. python /cdata/NT2C/evaluater/evaluate.py /cdata/pipeTemp/fineModel/model_epoch500.sav /cdata/thesis/pipeTemp_500_fineDenoise_27/ 
# ex. python /cdata/NT2C/evaluater/evaluate.py skip /cdata/thesis/topazGeneral/  (/cdata/thesis/topazGeneral/denoised/ should be filled with topaz denoised.)
########################################################################################

#input
raw_clean         = sys.argv[1]
raw_noisy         = sys.argv[2]
denoised          = sys.argv[3]
output_dir        = sys.argv[4]
skip_norm         = sys.argv[5]

# 0. Prepare Dirs.
out_root      = output_dir
out_wholeNorm = output_dir + "wholeNorm/"

if not os.path.exists(out_root):
    os.makedirs(out_root)
if not os.path.exists(out_wholeNorm):
    os.makedirs(out_wholeNorm)


# 1. Extract(with Normalize)
if skip_norm != 'true':
    normalize(denoised, out_wholeNorm)
print("---------------------------- Patch Norm Finished.")

# 3. metric evaluater
# calculate_metrics(raw_noisy, outputWHOLE_DIR, "whole")
calculate_pearsoncc(raw_clean, denoised, "pipeGauss")
print("---------------------------- Metric evaluation Finished. ")