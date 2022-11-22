import os
import sys
import glob
from math import floor, ceil
import mrcfile
from TARGET import back_patches, signal_patches
import numpy as np
from normalizer import normalize
# USAGE ################################################################################
# python raw_prepare.py [input_dir] [output_dir]
# ex. python /cdata/NT2C/evaluate/raw_prepare.py /cdata/thesis/raw/whole/ /cdata/thesis/raw/
########################################################################################

#input
input_dir        = sys.argv[1]
output_dir       = sys.argv[2]

cuda = 6
NT2C_DIR = "/cdata/NT2C/"

# 0. Prepare Dirs.
out_root      = output_dir
out_wholeNorm = output_dir + "wholeNorm/"
out_backNorm  = output_dir + "backNorm/"
out_signNorm  = output_dir + "signNorm/"

if not os.path.exists(out_root):
    os.makedirs(out_root)
if not os.path.exists(out_wholeNorm):
    os.makedirs(out_wholeNorm)
if not os.path.exists(out_backNorm):
    os.makedirs(out_backNorm)
if not os.path.exists(out_signNorm):
    os.makedirs(out_signNorm)

# 2. Extract(with Normalize)
input_DIR = input_dir
outputWHOLE_DIR= out_wholeNorm
outputBACK_DIR = out_backNorm
outputSIGN_DIR = out_signNorm

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

normalize(input_DIR, out_wholeNorm)

print("---------------------------- Patch Extract Finished.")