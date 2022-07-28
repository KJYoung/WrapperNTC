# python /cdata/NT2C/workUtil/fineVisualizer.py /cdata/benchDIR/60percent/fineModel/ /cdata/onlyTest/77-sb10-1-1_1.mrc /cdata/benchDIR/60percent/fineVisual/
import sys,os

modelDir  = sys.argv[1]  # 
srcPath   = sys.argv[2]  # 
outDir    = sys.argv[3]  # 

model_list= os.listdir(modelDir)

srcLastSlash  = srcPath.rindex('/')
srcFileName   = srcPath[srcLastSlash+1:]
print(srcFileName)
for i, modelName in enumerate(model_list, start=1):
    fineModelPath = modelDir + modelName
    #print(f'python /cdata/NT2C/denoiser/denoise_cmd.py -c 512 -o {outDir} -m {fineModelPath} -d 0 {srcPath}')
    status1 = os.system(f'python /cdata/NT2C/denoiser/denoise_cmd.py -c 512 -o {outDir} -m {fineModelPath} -d 0 {srcPath}')
    if status1 == 0:
        os.system(f'mv {outDir}{srcFileName} {outDir}{srcFileName[:-4]}_{modelName[:-4]}.mrc')
    else:
        quit()
print("All jobs done.")