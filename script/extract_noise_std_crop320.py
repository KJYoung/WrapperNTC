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
    #crop_1=(crop_1-np.min(crop_1))/(np.max(crop_1)-np.min(crop_1))
    #crop_2=(crop_2-np.min(crop_2))/(np.max(crop_2)-np.min(crop_2))
    #crop_3=(crop_3-np.min(crop_3))/(np.max(crop_3)-np.min(crop_3))
    #crop_4=(crop_4-np.min(crop_4))/(np.max(crop_4)-np.min(crop_4))
    
    # ssim12=ssim(crop_1,crop_2)
    # ssim13=ssim(crop_1,crop_3)
    # ssim14=ssim(crop_1,crop_4)
    # ssim23=ssim(crop_2,crop_3)
    # ssim24=ssim(crop_2,crop_4)
    # ssim34=ssim(crop_3,crop_4)
    # ssim_all=[ssim12,ssim13,ssim14,ssim23,ssim24,ssim34]
    # print('ssim12: {:.2f} | ssim13: {:.2f} | ssim14: {:.2f} | ssim23: {:.2f} | ssim24: {:.2f} | ssim34: {:.2f}'.format(ssim12,ssim13,ssim14,ssim23,ssim24,ssim34))
    # ssim_min=np.min(ssim_all)
    # if ssim_min<0.2:
    #     return False
    # else:
    #     print('ssim12: {:.2f} | ssim13: {:.2f} | ssim14: {:.2f} | ssim23: {:.2f} | ssim24: {:.2f} | ssim34: {:.2f}'.format(ssim12,ssim13,ssim14,ssim23,ssim24,ssim34))
    #     return True
    #std_global=np.std(arr)
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
denoised_dir=sys.argv[1]  #micrographs
raw_dir=sys.argv[2]  #raw micrographs
noise_dir=sys.argv[3]  #noise patches
draw_dir=sys.argv[4] #draw noise 

box_size=320
par_size=200
step=int(0.02*box_size)
#print('step:  {}'.format(step))
noise_patch_num=0
input_list=os.listdir(denoised_dir)
file_n = 1

import time 
script_start = time.time()

for input_file in input_list:
    file_start = time.time()
    print(os.path.join(denoised_dir,input_file))
    denoised_img=mrcfile.open(os.path.join(denoised_dir,input_file),permissive=True)
    denoised_img=denoised_img.data
    denoised_img=(denoised_img-np.min(denoised_img))/(np.max(denoised_img)-np.min(denoised_img))*255

    raw_img=mrcfile.open(os.path.join(raw_dir,input_file),permissive=True)
    raw_img=raw_img.data
    raw_img=(raw_img-np.min(raw_img))/(np.max(raw_img)-np.min(raw_img))*255
    #img = img.astype(np.uint8)
    
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
    if noise_patch_n != 0:
        with mrcfile.new(draw_noise_path,overwrite=True) as draw_noise:
            draw_noise.set_data(denoised_img)

    print(f"All jobs for {input_file} are done. : {file_n} out of {len(input_list)}")
    file_end = time.time()
    print(f"{file_end - file_start:.5f} sec for this file.")
    print(f"{file_end - script_start:.5f} sec so far for this file.")
    print(noise_patch_n, " out of ", patch_n)
    file_n += 1

print("---------------- extract_noise job is end ----------------")