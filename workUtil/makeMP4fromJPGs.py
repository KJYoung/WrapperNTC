def img2mp4(paths, pathOut , fps =10 ) :
    import cv2
    frame_array = []
    for idx , path in enumerate(paths) : 
        img = cv2.imread(path)
        height, width, layers = img.shape
        size = (width,height)
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

import os, sys
srcDir = sys.argv[1] 
outDir = sys.argv[2] 
paths = sorted(os.listdir(srcDir))
paths = sorted(os.listdir(srcDir), key = lambda x: int(x[x.rindex('_')+1:-4]))
# paths = sorted(os.listdir(srcDir), key = lambda x: int(x[x.rindex('_')+6:-4]))
print(paths)
paths = [os.path.join(srcDir,path) for path in paths]
img2mp4(paths , outDir, fps=3)

# mkdir jpgs && mv *.jpg jpgs && mkdir mrcs && mv *.mrc mrcs
# python /cdata/NT2C/workUtil/makeMP4fromJPGs.py /cdata/db1/other5lzf/fineVisual/jpgs/ /cdata/db1/other5lzf/fineVisual/5lzf100percentFPS3.mp4
# python /cdata/NT2C/workUtil/makeMP4fromJPGs.py /cdata/benchDIR/1o7percent/fineVisual/jpgs/ /cdata/benchDIR/1o7percent/fineVisual/5lzf1o7percent.mp4