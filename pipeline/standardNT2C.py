
from pipeline_config import parse_args_standard
import os, time
import numpy as np

def main(args):
    # 1. Coarse Denoiser Training
    pipelineStartTime   = time.time()
    NT2CDIR             = args.nt2cDir
    workspaceDIR        = args.workspace
    cleanDIR            = args.clean
    noisyDIR            = args.noisy
    cudaDevice          = args.device
    coarseEpochs        = args.coarseEpochs
    rawDataDIR          = args.raw
    extractCore         = args.extractCores
    extractDraw         = args.extractDraw
    coarseModel         = args.coarseModel
    coarseDenoised      = args.coarseDenoised
    noisePatch          = args.noisePatch
    loadGenerator       = args.load_G
    generatorIter       = args.generator_iters

    coarseSavePrefix    = workspaceDIR + 'coarseModel/' + 'model'
    coarseModelDIR      = workspaceDIR + 'coarseModel/'
    coarseDenoisedDIR   = workspaceDIR + 'coarseDenoised/'
    noisePatchDIR       = workspaceDIR + 'noisePatch/'
    noiseDrawDIR        = workspaceDIR + 'noiseDraw/'
    
    skipStep1           = False if (coarseModel == '') and (coarseDenoised == '') and (noisePatch == '') else True
    skipStep2           = False if (coarseDenoised == '') and (noisePatch == '') else True
    skipStep3           = False if (noisePatch == '') else True
    skipStep4           = False if (loadGenerator == '') else True

    print("--Step 1 : Coarse Denoiser Training! --")
    step1StartTime      = time.time()
    if skipStep1:
        print("---- Step 1 skipped.")
    else:
        if not os.path.exists(coarseModelDIR):      # directory for coarse Denoiser models.
            os.makedirs(coarseModelDIR)
        status1 = os.system(f'python {NT2CDIR}denoiser/denoise_cmd.py -a {noisyDIR} -b {cleanDIR} -d {cudaDevice} -c 800 --num-epochs {coarseEpochs} --save-prefix {coarseSavePrefix} > {workspaceDIR}coarseSTDOUT.log')
        if status1 != 0:
            print("----Step 1 was not successfully finished with error code {}".format(status1))
            quit()
        print("----Elapsed time for step 1 : {} seconds".format(time.time() - step1StartTime))
    # 2. Coarse Denoise Raw Micrographs
    print("--Step 2 : Coarse Denoising! --")
    step2StartTime      = time.time()
    if skipStep2:
        print("---- Step 2 skipped.")
    else:
        if not os.path.exists(coarseDenoisedDIR):   # directory for coarse denoised Raw micrographs.
            os.makedirs(coarseDenoisedDIR)
        
        if skipStep1:
            coarseModelPath = coarseModel
        else:
            digits = int(np.ceil(np.log10(coarseEpochs)))
            coarseModelPath = coarseSavePrefix + ('_epoch{:0'+str(digits)+'}.sav').format(coarseEpochs)
        status2 = os.system(f'python {NT2CDIR}denoiser/denoise_cmd.py -o {coarseDenoisedDIR} -m {coarseModelPath} -d {cudaDevice} {rawDataDIR}*.mrc')
        if status2 != 0:
            print("----Step 2 was not successfully finished with error code {}".format(status2))
            quit()
        print("----Elapsed time for step 2 : {} seconds".format(time.time() - step2StartTime))
    # 3. Noise extract.
    print("--Step 3 : Noise extraction! --")
    if skipStep3:
        print("---- Step 3 skipped.")
    else:
        step3StartTime      = time.time()
        if not os.path.exists(noisePatchDIR):       # directory for extracted noise patches.
            os.makedirs(noisePatchDIR)
        if extractDraw:
            if not os.path.exists(noiseDrawDIR):        # directory for visualization of extraction.
                os.makedirs(noiseDrawDIR)
        status3 = os.system(f'python {NT2CDIR}script/extract_noise_512_each.py {coarseDenoisedDIR} {rawDataDIR} {noisePatchDIR} {noiseDrawDIR} {extractCore} {extractDraw} 2> {workspaceDIR}noiseExtractSTDERR.log')
        if status3 != 0:
            print("----Step 3 was not successfully finished with error code {}".format(status3))
            quit()
        print("----Elapsed time for step 3 : {} seconds".format(time.time() - step3StartTime))
    # 3. GAN noise synthesizer training.
    print("--Step 4 : GAN noise synthesizer training! --")
    if skipStep4:
        print("---- Step 4 skipped.")
    else:
        step4StartTime      = time.time()
        if cudaDevice >= 0:
            os.system(f'CUDA_VISIBLE_DEVICES={cudaDevice} python {NT2CDIR}synthesizer/main_savemrc_512.py --batch_size 16 --dataroot {workspaceDIR}noisePatch/ --cuda True --generator_iters {generatorIter} > {workspaceDIR}GANtrain{generatorIter}.log')
        else:
            os.system(f'python {NT2CDIR}synthesizer/main_savemrc_512.py --batch_size 16 --dataroot {workspaceDIR}noisePatch/ --cuda False --generator_iters {generatorIter} > {workspaceDIR}GANtrain{generatorIter}.log')
        print("----Elapsed time for step 4 : {} seconds".format(time.time() - step4StartTime))

    # 4. GAN noise synthesize.
    
    print("Hi! Elapsed time : {} seconds".format(time.time() - pipelineStartTime))

if __name__ == '__main__':
    args = parse_args_standard()

    main(args)