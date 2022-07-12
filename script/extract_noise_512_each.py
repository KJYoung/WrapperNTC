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

def is_noise(arr,x0,x1,y0,y1):
    if (x0+x1)%2==0:
        x1=x1
    else:
        x1=x1-1
    if (y0+y1)%2==0:
        y1=y1
    else:
        y1=y1-1

    x_center=int(box_size*0.5)
    y_center=int(box_size*0.5)
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

denoised_dir    = sys.argv[1]       # micrographs
raw_dir         = sys.argv[2]       # raw micrographs
noise_dir       = sys.argv[3]       # noise patches
draw_dir        = sys.argv[4]       # draw noise 
worker          = int(sys.argv[5])  # number of workers.
extractDraw     = bool(sys.argv[6]) # whether save the noiseDraw or not.

box_size=512
step=int(0.02*box_size) 
noise_patch_num=0
input_list=os.listdir(denoised_dir)

script_start = time.time()

# Multi processing!
print('box_size : {}, step:  {}'.format(box_size, step))
# print("input length :", len(input_list))
jobsNUM = len(input_list) // worker
    
def processInputs(input_list):
    file_n = 1
    for input_file in input_list:
        file_start = time.time()
        print("Now extracting from ...", os.path.join(denoised_dir,input_file))
        denoised_img=mrcfile.open(os.path.join(denoised_dir,input_file),permissive=True)
        denoised_img=denoised_img.data
        denoised_img=(denoised_img-np.min(denoised_img))/(np.max(denoised_img)-np.min(denoised_img))*255

        raw_img=mrcfile.open(os.path.join(raw_dir,input_file),permissive=True)
        raw_img=raw_img.data
        raw_img=(raw_img-np.min(raw_img))/(np.max(raw_img)-np.min(raw_img))*255
        
        w,h=denoised_img.shape
        noise_patch_path=os.path.join(noise_dir,input_file)
        draw_noise_path=os.path.join(draw_dir,input_file)
        noise_patch_n=0
        patch_n=0
        for x in range(0,w,step):
            for y in range(0,h,step):
                if x+box_size>w or y+box_size>h:
                    continue
                denoised_arr=denoised_img[x:x+box_size,y:y+box_size]
                raw_arr=raw_img[x:x+box_size,y:y+box_size]
                patch_n += 1
                noise=is_noise(denoised_arr,x,x+box_size,y,y+box_size)
                if noise:  #save noise patch
                    with mrcfile.new(noise_patch_path[:-4]+'_'+str(noise_patch_n)+'.mrc',overwrite=True) as noise_patch:
                        noise_patch.set_data(raw_arr)
                    #draw noise patch location.
                    cv2.rectangle(denoised_img, (y,x), (y+box_size,x+box_size), (0, 0, 255), 2)
                    noise_patch_n+=1
        #save draw noise
        if noise_patch_n != 0 and (not extractDraw):
            with mrcfile.new(draw_noise_path,overwrite=True) as draw_noise:
                draw_noise.set_data(denoised_img)

        print(f"All jobs for {input_file} are done. : {file_n} out of {len(input_list)}")
        file_end = time.time()
        # print(f"{file_end - file_start:.5f} sec for this file.")
        eta_calc = (len(input_list) - file_n) * (file_end-script_start)/file_n
        print(f"{file_end - script_start:.5f} sec so far. eta : {eta_calc:.5f} sec [ ", noise_patch_n, " out of ", patch_n, " ]")
        file_n += 1

def mainTask(*input_list):
    # print("Input length : " , len(input_list), time.ctime())
    processInputs(list(input_list))

def main():
    threads = []
    for i in range(worker-1):
        print("worker ", i, " will process ", jobsNUM * i , " ~ " , jobsNUM * (i+1) - 1)
        T = multiprocessing.Process(target=mainTask, args=(input_list[jobsNUM * i : jobsNUM * (i+1)]))
        threads.append(T)
    print("worker ", i+1, " will process ", jobsNUM * (i+1) , " ~ " , len(input_list)-1)
    T = multiprocessing.Process(target=mainTask, args=(input_list[jobsNUM * (i+1) : ]))
    threads.append(T)

    for t in threads:
        t.start()

    for process in multiprocessing.active_children():
        process.join()

    for process in multiprocessing.active_children():
        print(process.name, process.is_alive())

if __name__ == "__main__":
    main()

print("---------------- extract_noise job is end ----------------")