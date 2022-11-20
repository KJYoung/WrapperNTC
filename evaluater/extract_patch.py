from math import floor, ceil
import mrcfile

input_DIR = ""
output_DIR = ""

patches = [
    # ['file name', 'patch x_start', 'patch x_end', 'patch y_start', 'patch y_end' ]
    { 
        'file' : 'sb1_210512 pos 10 1-1_1.mrc',
        'patches' : [
            [398.961, 1063.8961, 2792.72727, 3590.64935],
        ]
    }
]

for file in patches:
    orig = mrcfile.open(f"{input_DIR}{file['file']}", permissive=True).data
    for i, patch in enumerate(file['patches']):
        orig_ = orig.copy()
        orig_ = orig_[floor(patch[2]):ceil(patch[3]), floor(patch[0]):ceil(patch[1])]
        with mrcfile.new(f"{output_DIR}{file['file']}_{i}.mrc") as mrc:
            mrc.set_data(orig_)