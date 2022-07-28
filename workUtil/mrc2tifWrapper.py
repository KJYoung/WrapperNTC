# python /cdata/NT2C/workUtil/mrc2tifWrapper.py /cdata/db1/other5lzf/fineVisual/
import sys,os

srcDir  = sys.argv[1]  # 

input_list= os.listdir(srcDir)

for i, srcName in enumerate(input_list, start=1):
    status1 = os.system(f'mrc2tif -j {srcDir}{srcName} {srcDir}{srcName[:-4]}.jpg')
    if status1 != 0:
        quit()
print("All jobs done.")