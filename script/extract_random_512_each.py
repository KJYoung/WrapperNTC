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
import multiprocessing
import time 

raw_dir         = sys.argv[1]       # raw micrographs directory.
noise_dir       = sys.argv[2]       # result noise patches directory.
draw_dir        = sys.argv[3]       # (debug) draw noise 
denoised_dir    = sys.argv[4]       # (debug) denoised micrographs
worker          = int(sys.argv[5])  # number of workers.
targetNum       = int(sys.argv[6])  # target number of result patches.

box_size=512
step=int(0.02*box_size) 
noise_patch_num=0
input_list=os.listdir(raw_dir)

jobNumList     =  np.full(worker, targetNum//worker)
jobNumList[-1] += targetNum - np.sum(jobNumList)
script_start = time.time()

jobsNUM = len(input_list) // worker

assert targetNum == np.sum(jobNumList)

    
def processInputs(id, input_list):
    noise_patch_n = 0
    noise_patch_n_target = jobNumList[id]
    noise_patch_per_img  = (noise_patch_n_target // len(input_list)) + 1 # 717 / 384 = 1.84 => 2 per img.
    print("ID : {} / Target number : {} / First Target : {} / Per img {}".format(id, noise_patch_n_target, input_list[0], noise_patch_per_img))
    # 717개 해야하는데, input은 384

    file_n = 1
    for input_file in input_list:
        # print("Now extracting from ...", os.path.join(raw_dir,input_file))
        denoised_img=mrcfile.open(os.path.join(denoised_dir,input_file),permissive=True)
        denoised_img=denoised_img.data
        denoised_img=(denoised_img-np.min(denoised_img))/(np.max(denoised_img)-np.min(denoised_img))*255

        raw_img=mrcfile.open(os.path.join(raw_dir,input_file),permissive=True)
        raw_img=raw_img.data
        raw_img=(raw_img-np.min(raw_img))/(np.max(raw_img)-np.min(raw_img))*255
        w,h=raw_img.shape

        noise_patch_path=os.path.join(noise_dir,input_file)
        draw_noise_path=os.path.join(draw_dir,input_file)

        if noise_patch_n >= noise_patch_n_target:
            break
        
        patch_n = 1
        for k in range(noise_patch_per_img):
            x = np.random.randint(0, w-box_size)
            y = np.random.randint(0, h-box_size)
            assert x+box_size<=w or y+box_size<=h
            
            raw_arr=raw_img[x:x+box_size,y:y+box_size]
            with mrcfile.new(noise_patch_path[:-4]+'_'+str(patch_n)+'.mrc',overwrite=True) as noise_patch:
                noise_patch.set_data(raw_arr)
            cv2.rectangle(denoised_img, (y,x), (y+box_size,x+box_size), (0, 0, 255), 2)
            noise_patch_n += 1
            patch_n += 1
        
        if patch_n != 1:
            with mrcfile.new(draw_noise_path,overwrite=True) as draw_noise:
                draw_noise.set_data(denoised_img)
        print(f"All jobs for {input_file} are done. : {file_n} out of {len(input_list)}")
        file_end = time.time()
        eta_calc = (len(input_list) - file_n) * (file_end-script_start)/file_n
        print(f"{file_end - script_start:.5f} sec so far. eta : {eta_calc:.5f} sec")
        file_n += 1

def print_sum(*input_list):
    processInputs(input_list[0], input_list[1])

if __name__ == "__main__":
    threads = []
    for i in range(worker-1):
        print("worker ", i, " will process ", jobsNUM * i , " ~ " , jobsNUM * (i+1) - 1)
        T = multiprocessing.Process(target=print_sum, args=(i, input_list[jobsNUM * i : jobsNUM * (i+1)]))
        threads.append(T)
    print("worker ", i+1, " will process ", jobsNUM * (i+1) , " ~ " , len(input_list)-1)
    T = multiprocessing.Process(target=print_sum, args=(i+1, input_list[jobsNUM * (i+1) : ]))
    threads.append(T)

    for t in threads:
        t.start()

    for process in multiprocessing.active_children():
        process.join()

    for process in multiprocessing.active_children():
        print(process.name, process.is_alive())