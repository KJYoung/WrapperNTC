# from skimage.measure import compare_ssim as ssim
# from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from matplotlib import pyplot as plt
import mrcfile
import mahotas
import sys,os
import matplotlib.pyplot as plt
import pywt
import time
import multiprocessing
import argparse

def is_noise(arr,x0,x1,y0,y1):
    if (x0+x1)%2==0:
        x1=x1
    else:
        x1=x1-1
    if (y0+y1)%2==0:
        y1=y1
    else:
        y1=y1-1

    x_center=int(extractSize*0.5)
    y_center=int(extractSize*0.5)
    # print("x0:{}|x1:{}|y0:{}|y1:{}".format(x0,x1,y0,y1))
    crop_1=arr[:y_center,:x_center]
    crop_2=arr[:y_center,x_center:]
    crop_3=arr[y_center:,:x_center]
    crop_4=arr[y_center:,x_center:]
    std_global = 10.0 #std_global=np.std(arr)
    std_1=np.std(crop_1)
    std_2=np.std(crop_2)
    std_3=np.std(crop_3)
    std_4=np.std(crop_4)
    # print('std_global: {:.4f} | std_1: {:.4f} | std_2: {:.4f} | std_3: {:.4f} | std_4: {:.4f} '.format(std_global,std_1,std_2,std_3,std_4))

    isnoise=True
    if std_1>0.1*std_global:
        isnoise=False
        return isnoise
    if std_2>0.1*std_global:
        isnoise=False
        return isnoise
    if std_3>0.1*std_global:
        isnoise=False
        return isnoise
    if std_4>0.1*std_global:
        isnoise=False
        return isnoise
    # print('True :: std_global: {} | std_1: {} | std_2: {} | std_3: {} | std_4: {} '.format(std_global,std_1,std_2,std_3,std_4))
    return isnoise

def parse_args_extract():
    parser = argparse.ArgumentParser(description="Noise extraction based on coarse denoised samples.")

    # directories
    parser.add_argument('--denoised',   type=str, required=True, help='denoised micrographs directory.')
    parser.add_argument('--raw',        type=str, required=True, help='raw micrographs directory.')
    parser.add_argument('--noisePatch', type=str, required=True, help='extracted noise Patches target directory.')
    parser.add_argument('--noiseDraw',  type=str, default='', help='visualization of noise Patches directory.')
    
    parser.add_argument('--worker',     type=int, default=1, help='number of workers')
    parser.add_argument('-s', '--size', type=int, default=512, help='extraction size')

    parser.add_argument('-n', '--num',  type=int, default=10000, help='number of patches to be extracted(random, randgauss workflow).')
    return parser.parse_args()

args = parse_args_extract()
denoised_dir    = args.denoised
raw_dir         = args.raw
noisePatchDIR   = args.noisePatch
draw_dir        = args.noiseDraw    # empty string '' if we don't want to generate noiseDraw.
extractSize     = args.size
worker          = args.worker
extractDraw     = False if draw_dir == '' else True

step            = int(0.02*extractSize) 
noise_patch_num = 0
input_list      = os.listdir(denoised_dir)

script_start    = time.time()

print('extractSize : {}, step:  {}'.format(extractSize, step))
jobsNUM = len(input_list) // worker
    
def processInputs(input_list):
    for input_file in input_list:
        denoised_img=mrcfile.open(os.path.join(denoised_dir,input_file),permissive=True)
        denoised_img=denoised_img.data
        denoised_img=(denoised_img-np.min(denoised_img))/(np.max(denoised_img)-np.min(denoised_img))*255

        raw_img=mrcfile.open(os.path.join(raw_dir,input_file),permissive=True)
        raw_img=raw_img.data
        raw_img=(raw_img-np.min(raw_img))/(np.max(raw_img)-np.min(raw_img))*255
        
        w,h=denoised_img.shape
        noise_patch_path=os.path.join(noisePatchDIR, input_file)
        draw_noise_path=os.path.join(draw_dir, input_file)
        noise_patch_n=0
        patch_n=0
        for x in range(0,w,step):
            for y in range(0,h,step):
                if x+extractSize>w or y+extractSize>h:
                    continue
                denoised_arr=denoised_img[x:x+extractSize,y:y+extractSize]
                raw_arr=raw_img[x:x+extractSize,y:y+extractSize]
                patch_n += 1
                noise=is_noise(denoised_arr,x,x+extractSize,y,y+extractSize)
                if noise:  #save noise patch
                    with mrcfile.new(noise_patch_path[:-4]+'_'+str(noise_patch_n)+'.mrc',overwrite=True) as noise_patch:
                        noise_patch.set_data(raw_arr)
                    
                    if extractDraw: # draw noise patch location.
                        cv2.rectangle(denoised_img, (y,x), (y+extractSize,x+extractSize), (0, 0, 255), 2)
                    noise_patch_n+=1
        if noise_patch_n != 0 and extractDraw: # save draw noise
            with mrcfile.new(draw_noise_path,overwrite=True) as draw_noise:
                draw_noise.set_data(denoised_img)

def processInputsReport(input_list):
    for i, input_file in enumerate(input_list, start=1):
        file_start = time.time()
        denoised_img=mrcfile.open(os.path.join(denoised_dir,input_file),permissive=True)
        denoised_img=denoised_img.data
        denoised_img=(denoised_img-np.min(denoised_img))/(np.max(denoised_img)-np.min(denoised_img))*255

        raw_img=mrcfile.open(os.path.join(raw_dir,input_file),permissive=True)
        raw_img=raw_img.data
        raw_img=(raw_img-np.min(raw_img))/(np.max(raw_img)-np.min(raw_img))*255
        
        w,h=denoised_img.shape
        noise_patch_path=os.path.join(noisePatchDIR,input_file)
        draw_noise_path=os.path.join(draw_dir,input_file)
        noise_patch_n=0
        patch_n=0
        for x in range(0,w,step):
            for y in range(0,h,step):
                if x+extractSize>w or y+extractSize>h:
                    continue
                denoised_arr=denoised_img[x:x+extractSize,y:y+extractSize]
                raw_arr=raw_img[x:x+extractSize,y:y+extractSize]
                patch_n += 1
                noise=is_noise(denoised_arr,x,x+extractSize,y,y+extractSize)
                if noise:  #save noise patch
                    with mrcfile.new(noise_patch_path[:-4]+'_'+str(noise_patch_n)+'.mrc',overwrite=True) as noise_patch:
                        noise_patch.set_data(raw_arr)
                    if extractDraw: # draw noise patch location.
                        cv2.rectangle(denoised_img, (y,x), (y+extractSize,x+extractSize), (0, 0, 255), 2)
                    noise_patch_n+=1
        
        if noise_patch_n != 0 and extractDraw: # save draw noise
            with mrcfile.new(draw_noise_path,overwrite=True) as draw_noise:
                draw_noise.set_data(denoised_img)

        file_end = time.time()
        eta_calc = (len(input_list) - i) * (file_end-script_start)/i
        print('# Noise extract in representative thread : [{}/{}] {:.2%} || {:.3f}s/file [{} extracted] || {:.4f} so far || eta : {:.4f}'.format(i, len(input_list), i/len(input_list),
                                                                                                                                                file_end - file_start, noise_patch_n,
                                                                                                                                                file_end - script_start, eta_calc), file=sys.stdout, end='\r')

def mainTask(*input_list):
    processInputs(list(input_list))

def mainTaskReport(*input_list):
    processInputsReport(list(input_list))

if __name__ == "__main__":
    if worker > 1: # Multithreading
        threads = []
        for i in range(worker-1):
            print("worker ", i, " will process ", jobsNUM * i , " ~ " , jobsNUM * (i+1) - 1)
            T = multiprocessing.Process(target=mainTask, args=(input_list[jobsNUM * i : jobsNUM * (i+1)]))
            threads.append(T)
        print("worker ", i+1, " will process ", jobsNUM * (i+1) , " ~ " , len(input_list)-1)
        T = multiprocessing.Process(target=mainTaskReport, args=(input_list[jobsNUM * (i+1) : ]))
        threads.append(T)

        for t in threads:
            t.start()

        for process in multiprocessing.active_children():
            process.join()

        for process in multiprocessing.active_children():
            print(process.name, process.is_alive())
    elif worker == 1:
        processInputsReport(input_list)
    else:
        print("Worker should be >= 1.")
        assert False