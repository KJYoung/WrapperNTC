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
# python evaluateSIM.py [model_path] [output_dir] [skip_to_go_metric]
# ex. Evaluate PSNR on Already normalized micrographs
# python /cdata/NT2C/evaluater/evaluateSIM.py skip /cdata/thesis/simNorm/2wrj/noisyNorm/ /cdata/thesis/simEval/epoch0/ /cdata/thesis/simEval/epoch0/ true
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
calculate_metrics(raw_noisy, out_wholeNorm, "whole")
# calculate_pearsoncc(raw_clean, denoised, "topazGeneral")
print("---------------------------- Metric evaluation Finished. ")