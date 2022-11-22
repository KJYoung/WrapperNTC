import os
import glob
import mrcfile
import numpy as np

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