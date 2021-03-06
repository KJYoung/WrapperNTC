# -*- coding:utf-8 -*
import numpy as np
from scipy.optimize import leastsq
import sys,os
import cv2
import operator
import mrcfile
import random

from functools import reduce

def func(p,x,y,noise_mean):
    k1,k2,b=p
    return k1*x+k2*(y-noise_mean)+b

def error(p,x,y,noise_mean,z):
    #print(s)
    return func(p,x,y,noise_mean)-z 

def fit_noise(clean,noise,noisy):
    # print(noise.shape)
    # noise=noise[0,:,:]
    Xi=np.array([i for j in clean for i in j],dtype=np.float64)
    Yi=np.array([i for j in noise for i in j],dtype=np.float64)
    Zi=np.array([i for j in noisy for i in j],dtype=np.float64)
    noise_mean=np.mean(Yi)
    p0=[100,100,2]
    
    # (4096, 4096) (128, 128) (4096, 4096) ()
    Xi = Xi.flatten()
    Yi = Yi.flatten()
    Zi = Zi.flatten()
    # print(Xi.shape, Yi.shape, Zi.shape, noise_mean.shape)
    # (16384,) (16384,) (16384,) ()
    # print(Xi.dtype, Yi.dtype, Zi.dtype, noise_mean.dtype)
    Para=leastsq(error,p0,args=(Xi,Yi,noise_mean,Zi))
    k1,k2,b=Para[0]

    return k1,k2,b
    
clean_dir=sys.argv[1]
noise_dir=sys.argv[2]
noisy_dir=sys.argv[3]
noisy_gen_dir=sys.argv[4]

clean_list =os.listdir(clean_dir)
noise_list=os.listdir(noise_dir)
noisy_list=os.listdir(noisy_dir)
noise_len=len(noise_list)

for clean_img in clean_list:
    for reIndex in range(3): # 64000 * 3 = 192000
        #assert False
        rand1=random.randint(0,noise_len-1)
        print(noisy_gen_dir+clean_img[:-4]+"_{}".format(reIndex)+".mrc" + " | with noise {}".format(rand1))
        # rand1=0 # Fix for merge. ##################################################
        #clean_path=clean_dir+clean_img
        #noisy_path=noisy_dir+clean_img
        #noise_path=noise_dir+noise_list[rand1]
        clean_path=os.path.join(clean_dir,clean_img)
        noisy_path=os.path.join(noisy_dir,clean_img)
        noise_path=os.path.join(noise_dir,noise_list[rand1])

        clean=mrcfile.open(clean_path)
        noisy=mrcfile.open(noisy_path)
        noise=mrcfile.open(noise_path)
        clean=clean.data
        noisy=noisy.data
        noise=noise.data
        clean=(clean-np.min(clean))/(np.max(clean)-np.min(clean))*255.0
        noisy=(noisy-np.min(noisy))/(np.max(noisy)-np.min(noisy))*255.0
        noise=(noise-np.min(noise))/(np.max(noise)-np.min(noise))*255.0
        k1,k2,b=fit_noise(clean,noise,noisy)
        # print('k1,k2,b: {}, {}, {}'.format(k1,k2,b))
        noisy_gen=k1*clean+k2*(noise-np.mean(noise))+b+(noise-np.mean(noise))
        #noisy_gen = noisy_gen.reshape((640,640))
        #noisy_gen = noisy_gen.reshape((320,320))
        # (1, 128, 128) -> (128, 128)
        noisy_gen = noisy_gen.reshape((noisy_gen.shape[1],noisy_gen.shape[2]))
        # print("noisy_gen : ", noisy_gen.shape)
        fit_noisy_out=mrcfile.new(noisy_gen_dir+clean_img[:-4]+"_{}".format(reIndex)+".mrc",overwrite=True)
        fit_noisy_out.set_data(noisy_gen.astype(np.float32))

# This file add_norm_noise_128 for now.
# python3 add_norm_noise_320.py /cdata/temp/clean/ /cdata/NT2C/noise_synthesizer/pytorch-wgan-master/synthesized_noises/ /cdata/temp/noisy/  /cdata/temp/reweighted/
# python3 add_norm_noise_320.py /cdata/db1/fragmentTEM/clean4/ /cdata/NT2C/noise_synthesizer/pytorch-wgan-master/synthesized_noises/ /cdata/db1/fragmentTEM/noisy4/  /cdata/db1/noiseReweight/
