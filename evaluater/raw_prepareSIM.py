import os
import sys
import glob
from math import floor, ceil
import mrcfile
import numpy as np

# USAGE ################################################################################
# python raw_prepareSIM.py [noisy_dir] [clean_dir] [output_dir]
# ex. python /cdata/NT2C/evaluater/raw_prepareSIM.py /cdata/WrapperTEM/Micrographs/noisy4/ /cdata/WrapperTEM/Micrographs/clean4/ /cdata/thesis/simNorm/2wrj/
########################################################################################

#input
noisy_dir        = sys.argv[1]
clean_dir        = sys.argv[2]
output_dir       = sys.argv[3]

# 0. Prepare Dirs.
out_root      = output_dir
out_cleanNorm = output_dir + "cleanNorm/"
out_noisyNorm = output_dir + "noisyNorm/"

if not os.path.exists(out_root):
    os.makedirs(out_root)
if not os.path.exists(out_cleanNorm):
    os.makedirs(out_cleanNorm)
if not os.path.exists(out_noisyNorm):
    os.makedirs(out_noisyNorm)

# 2. Normalize
def normalize(input, output):
    A = []
    B = []
    for path in glob.glob(input + '*.mrc'):
        name = os.path.basename(path)
        A.append(path)
        B.append(output + name)

    for ins, outs in zip(A, B):
        orig = mrcfile.open(f"{ins}", permissive=True).data.copy()
        with mrcfile.new(f"{outs}") as mrc:
            mrc.set_data(orig / np.max(orig))

normalize(noisy_dir, out_noisyNorm)
normalize(clean_dir, out_cleanNorm)

print("---------------------------- Normalization Finished.")