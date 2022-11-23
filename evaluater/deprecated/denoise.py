import os

os.system("python /cdata/NT2C/denoiser/denoise_cmd.py -o /cdata/thesis/simEval/reduced5lzf10/ -c 512 -m /cdata/benchDIR/10percent/fineModel/model_epoch300.sav -d 7 /cdata/WrapperTEM/Micrographs/noisy4/*.mrc")

os.system("python /cdata/NT2C/denoiser/denoise_cmd.py -o /cdata/thesis/simEval/reduced5lzf3/ -c 512 -m /cdata/benchDIR/3percent/fineModel/model_epoch300.sav -d 7 /cdata/WrapperTEM/Micrographs/noisy4/*.mrc")

os.system("python /cdata/NT2C/denoiser/denoise_cmd.py -o /cdata/thesis/simEval/reduced5lzf31o7/ -c 512 -m /cdata/benchDIR/1o7percent/fineModel/model_epoch300.sav -d 7 /cdata/WrapperTEM/Micrographs/noisy4/*.mrc")

os.system("python /cdata/NT2C/denoiser/denoise_cmd.py -o /cdata/thesis/simEval/less2/ -c 512 -m /cdata/benchDIR/less2/fineModel/model_epoch100.sav -d 7 /cdata/WrapperTEM/Micrographs/noisy4/*.mrc")

os.system("python /cdata/NT2C/denoiser/denoise_cmd.py -o /cdata/thesis/simEval/less7/ -c 512 -m /cdata/benchDIR/less7/fineModel/model_epoch50.sav -d 7 /cdata/WrapperTEM/Micrographs/noisy4/*.mrc")

os.system("python /cdata/NT2C/denoiser/denoise_cmd.py -o /cdata/thesis/simEval/less8/ -c 512 -m /cdata/benchDIR/less8/fineModel/model_epoch50.sav -d 7 /cdata/WrapperTEM/Micrographs/noisy4/*.mrc")

os.system("python /cdata/NT2C/denoiser/denoise_cmd.py -o /cdata/thesis/simEval/less9/ -c 512 -m /cdata/benchDIR/less9/fineModel/model_epoch50.sav -d 7 /cdata/WrapperTEM/Micrographs/noisy4/*.mrc")

os.system("python /cdata/NT2C/denoiser/denoise_cmd.py -o /cdata/thesis/simEval/less10/ -c 512 -m /cdata/benchDIR/less10/fineModel/model_epoch100.sav -d 7 /cdata/WrapperTEM/Micrographs/noisy4/*.mrc")

os.system("python /cdata/NT2C/denoiser/denoise_cmd.py -o /cdata/thesis/simEval/less13/ -c 512 -m /cdata/benchDIR/less13/fineModel/model_epoch100.sav -d 7 /cdata/WrapperTEM/Micrographs/noisy4/*.mrc")

os.system("python /cdata/NT2C/denoiser/denoise_cmd.py -o /cdata/thesis/simEval/less15/ -c 512 -m /cdata/benchDIR/less15/fineModel/model_epoch100.sav -d 7 /cdata/WrapperTEM/Micrographs/noisy4/*.mrc")

os.system("python /cdata/NT2C/denoiser/denoise_cmd.py -o /cdata/thesis/simEval/less17/ -c 512 -m /cdata/benchDIR/less17/fineModel/model_epoch100.sav -d 7 /cdata/WrapperTEM/Micrographs/noisy4/*.mrc")

os.system("python /cdata/NT2C/denoiser/denoise_cmd.py -o /cdata/thesis/simEval/epoch0/ -c 512 -d 7 /cdata/WrapperTEM/Micrographs/noisy4/*.mrc")