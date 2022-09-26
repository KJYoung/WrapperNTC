# from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
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

def parse_args_extract():
    parser = argparse.ArgumentParser(description="Noise extraction based on coarse denoised samples.")

    # directories
    parser.add_argument('--denoised',   type=str, help='(DEBUGGING) denoised micrographs directory.')
    parser.add_argument('--raw',        type=str, required=True, help='raw micrographs directory.')
    parser.add_argument('--noisePatch', type=str, required=True, help='extracted noise Patches target directory.')
    parser.add_argument('--noiseDraw',  type=str, default='', help='(DEBUGGING) visualization of noise Patches directory.')
    
    parser.add_argument('--worker',     type=int, default=1, help='number of workers')
    parser.add_argument('-s', '--size', type=int, default=512, help='extraction size')
    parser.add_argument('-a', '--aug',  type=int, default=5, help='stochastic augmentation')

    parser.add_argument('-n', '--num',  type=int, default=10000, help='number of patches to be extracted(random, randgauss workflow).')
    return parser.parse_args()

args = parse_args_extract()
denoised_dir    = args.denoised
raw_dir         = args.raw
noisePatchDIR   = args.noisePatch
draw_dir        = args.noiseDraw    # empty string '' if we don't want to generate noiseDraw.
extractSize     = args.size
worker          = args.worker
aug             = args.aug
extractDraw     = False if draw_dir == '' else True

assert not (extractDraw == False and denoised_dir == "")
    
targetNum       = args.num          # target number of result patches.

step            = int(0.02*extractSize) 
noise_patch_num = 0
##input_list      = os.listdir(denoised_dir)
input_list=os.listdir(raw_dir)
jobNumList     =  np.full(worker, targetNum//worker)
jobNumList[-1] += targetNum - np.sum(jobNumList)

script_start    = time.time()

print('extractSize : {}, step:  {}'.format(extractSize, step))
jobsNUM = len(input_list) // worker
    
def processInputs(id, input_list):
    noise_patch_n = 0
    noise_patch_n_target = jobNumList[id]
    noise_patch_per_img  = (noise_patch_n_target // len(input_list)) + 1

    for i, input_file in enumerate(input_list, start=1):
        if extractDraw:
            denoised_img=mrcfile.open(os.path.join(denoised_dir,input_file),permissive=True)
            denoised_img=denoised_img.data
            denoised_img=(denoised_img-np.min(denoised_img))/(np.max(denoised_img)-np.min(denoised_img))*255
            denoised_img_stock=mrcfile.open(os.path.join(denoised_dir,input_file),permissive=True)
            denoised_img_stock=denoised_img_stock.data
            denoised_img_stock=(denoised_img_stock-np.min(denoised_img_stock))/(np.max(denoised_img_stock)-np.min(denoised_img_stock))*255

        raw_img=mrcfile.open(os.path.join(raw_dir,input_file),permissive=True)
        raw_img=raw_img.data
        try:
            raw_img=(raw_img-np.min(raw_img))/(np.max(raw_img)-np.min(raw_img))*255
        except TypeError:
            print("Warning : Due to typeError, the resulting patch number will be reduced.")
            continue
        
        w,h=raw_img.shape
        noise_patch_path=os.path.join(noisePatchDIR,input_file)
        draw_noise_path=os.path.join(draw_dir,input_file)
        draw_noise_path_stoc=os.path.join(draw_dir,f"STOCHASTIC_{input_file}")
        
        if noise_patch_n >= noise_patch_n_target:
            break
        
        patch_candida = []
        for patch_n, k in enumerate(range(aug * noise_patch_per_img)):
            x = np.random.randint(0, w-extractSize)
            y = np.random.randint(0, h-extractSize)
            assert x+extractSize<=w or y+extractSize<=h
            
            raw_arr=raw_img[x:x+extractSize, y:y+extractSize].astype('float32')
            center_coord=int(extractSize*0.5)
            std_1=np.std(raw_arr[:center_coord,:center_coord])
            std_2=np.std(raw_arr[:center_coord,center_coord:])
            std_3=np.std(raw_arr[center_coord:,:center_coord])
            std_4=np.std(raw_arr[center_coord:,center_coord:])
            max_std = max(std_1, std_2, std_3, std_4)
            patch_candida.append((raw_arr, max_std, std_1, std_2, std_3, std_4, x, y))
            if extractDraw: # draw noise patch location.
                cv2.rectangle(denoised_img, (y,x), (y+extractSize,x+extractSize), (0, 0, 255), 2)
        patch_candida.sort(key=lambda a: a[1]) # Ascending Sort for max_std.
        patch_candida = patch_candida[:noise_patch_per_img]
        # print(patch_candida[0][1], patch_candida[-1][1])
        for i, patch in enumerate(patch_candida):
            with mrcfile.new(noise_patch_path[:-4]+'_'+str(i)+'.mrc',overwrite=True) as noise_patch:
                noise_patch.set_data(patch[0])
            noise_patch_n += 1
            if extractDraw: # draw noise patch location.
                y, x = patch[7], patch[6]
                cv2.rectangle(denoised_img_stock, (y,x), (y+extractSize,x+extractSize), (0, 0, 255), 2)
            if noise_patch_n == noise_patch_n_target:
                break
            # noise_patch_n += 1
        if extractDraw:
            with mrcfile.new(draw_noise_path,overwrite=True) as draw_noise:
                draw_noise.set_data(denoised_img)
            with mrcfile.new(draw_noise_path_stoc,overwrite=False) as draw_noise:
                draw_noise.set_data(denoised_img_stock)
            
def mainTask(*input_list):
    processInputs(input_list[0], input_list[1])

def mainTaskReport(*input_list):
    processInputs(input_list[0], input_list[1])

if __name__ == "__main__":
    if worker > 1: # Multithreading
        threads = []
        for i in range(worker-1):
            print("worker ", i, " will process ", jobsNUM * i , " ~ " , jobsNUM * (i+1) - 1)
            T = multiprocessing.Process(target=mainTask, args=(i, input_list[jobsNUM * i : jobsNUM * (i+1)]))
            threads.append(T)
        print("worker ", i+1, " will process ", jobsNUM * (i+1) , " ~ " , len(input_list)-1)
        T = multiprocessing.Process(target=mainTaskReport, args=(i+1, input_list[jobsNUM * (i+1) : ]))
        threads.append(T)

        for t in threads:
            t.start()

        for process in multiprocessing.active_children():
            process.join()

        for process in multiprocessing.active_children():
            print(process.name, process.is_alive())
    elif worker == 1:
        processInputs(0, input_list)
    else:
        print("Worker should be >= 1.")
        assert False