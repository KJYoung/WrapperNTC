from pipeline_config import parse_args_standard
import os, time
import numpy as np

def step1CoarseTrain():
    if not os.path.exists(coarseModelDIR):      # directory for coarse denoiser models.
        os.makedirs(coarseModelDIR)
    status1 = os.system(f'python {NT2CDIR}denoiser/denoise_cmd.py -a {noisyDIR} -b {cleanDIR} -d {cudaDevice} -c 800 --num-epochs {coarseEpochs} --save-prefix {coarseSavePrefix} > {workspaceDIR}coarseSTDOUT.log')
    if status1 != 0:
        print("----Step 1 was not successfully finished with error code {}".format(status1))
        quit()

def step2CoraseDenoise(skipStep1):
    if not os.path.exists(coarseDenoisedDIR):   # directory for coarse denoised Raw micrographs.
        os.makedirs(coarseDenoisedDIR)
    
    if skipStep1:
        coarseModelPath = coarseModel
    else:
        digits = int(np.ceil(np.log10(coarseEpochs)))
        coarseModelPath = coarseSavePrefix + ('_epoch{:0'+str(digits)+'}.sav').format(coarseEpochs)
    
    patchSizeParam = "" if patchSize == -1 else f"-s {patchSize}"
    status2 = os.system(f'python {NT2CDIR}denoiser/denoise_cmd.py -o {coarseDenoisedDIR} -m {coarseModelPath} -d {cudaDevice} {patchSizeParam} {rawDataDIR}*.mrc')
    
    if status2 != 0:
        print("----Step 2 was not successfully finished with error code {}".format(status2))
        quit()

def step3NoiseExtract():
    if not os.path.exists(noisePatchDIR):       # directory for extracted noise patches.
        os.makedirs(noisePatchDIR)
    if extractDraw:
        if not os.path.exists(noiseDrawDIR):    # directory for visualization of extraction.
            os.makedirs(noiseDrawDIR)

    noiseDrawOption = '' if extractDraw == False else f'--noiseDraw {noiseDrawDIR}'
    status3 = os.system(f'python {NT2CDIR}script/extractScript.py --denoised {coarseDenoisedDIR} --raw {rawDataDIR} --noisePatch {noisePatchDIR} {noiseDrawOption} --worker {extractCore} -s 512 2> {workspaceDIR}noiseExtractSTDERR.log')
    if status3 != 0:
        print("----Step 3 was not successfully finished with error code {}".format(status3))
        quit()

def step4GANtrain():
    GANtrainDataDIR = (workspaceDIR + "noisePatch/") if noisePatch == "" else noisePatch
    trainGridParam = "--synGrid" if trainGrid else ""
    CUDA_ENV_Param = f"CUDA_VISIBLE_DEVICES={cudaDevice}" if cudaDevice >= 0 else ""
    cudaParam      = "True" if cudaDevice >= 0 else "False"

    status4 = os.system(f'{CUDA_ENV_Param} python {NT2CDIR}synthesizer/main_savemrc_512.py --batch_size 16 --dataroot {GANtrainDataDIR} --cuda {cudaParam} --generator_iters {generatorIter} --output_dir {workspaceDIR} {trainGridParam} > {workspaceDIR}GANtrain{generatorIter}.log 2> {workspaceDIR}GANtrainSTDERR.log')
        
    if status4 != 0:
        print("----Step 4 was not successfully finished with error code {}".format(status4))
        quit()

def step5GANsynthesize():
    synGridParam = "--synGrid" if synGrid else ""
    CUDA_ENV_Param = f"CUDA_VISIBLE_DEVICES={cudaDevice}" if cudaDevice >= 0 else ""
    cudaParam      = "True" if cudaDevice >= 0 else "False"

    status5 = os.system(f'{CUDA_ENV_Param} python {NT2CDIR}synthesizer/main_generateV_512.py --batch_size 64 --dataroot / --cuda {cudaParam} --output_dir {workspaceDIR} --load_G {generatorPath} --synNum64 {synNum64} {synGridParam} 2> /dev/null')
        
    if status5 != 0:
        print("----Step 5 was not successfully finished with error code {}".format(status5))
        quit()

def step6FragmentReweight():
    if fragmentNoisy == '': # Fragment Noisy
        if not os.path.exists(fragmentNoisyDIR):       # directory for fragmented noisy patches.
            os.makedirs(fragmentNoisyDIR)
        status6_1 = os.system(f'python {NT2CDIR}workUtil/fragmentWrapper.py -s 512 -d {noisyDIR} -id {fragmentNoisyDIR}')
        if status6_1 != 0:
            print("----Step 6-1 : Noisy Fragmentation was not successfully finished with error code {}".format(status6_1))
            quit()
    
    if fragmentClean == '': # Fragment Clean
        if not os.path.exists(fragmentCleanDIR):       # directory for fragmented clean patches.
            os.makedirs(fragmentCleanDIR)
        status6_2 = os.system(f'python {NT2CDIR}workUtil/fragmentWrapper.py -s 512 -d {cleanDIR} -id {fragmentCleanDIR}')
        if status6_2 != 0:
            print("----Step 6-2 : Clean Fragmentation was not successfully finished with error code {}".format(status6_2))
            quit()
    
    if noiseReweight == '': # Noise reweight
        if not os.path.exists(noiseReweightDIR):
            os.makedirs(noiseReweightDIR)
        status6_3 = os.system(f'python {NT2CDIR}script/noise_reweight.py {fragmentCleanDIR} {synNoiseDIR} {fragmentNoisyDIR} {noiseReweightDIR} {augNum}')
        if status6_3 != 0:
            print("----Step 6-3 : Noise reweighting was not successfully finished with error code {}".format(status6_3))
            quit()

def step7Finetrain():
    if not os.path.exists(fineModelDIR):      # directory for fine Denoiser models.
        os.makedirs(fineModelDIR)
    status7_2 = os.system(f'python {NT2CDIR}denoiser/denoise_cmd.py -a {noiseReweightDIR} -b {fragmentCleanDIR} -d {cudaDevice} -c 512 --num-epochs {fineEpochs} --lr {fineLR} --batch-size {fineBatch} --save-prefix {fineSavePrefix} --augmented > {workspaceDIR}fineSTDOUT.log')
    if status7_2 != 0:
        print("----Step 7-2 : Fine denoiser training was not successfully finished with error code {}".format(status7_2))
        return -1
    return 0

def step8FineDenoise(skipStep7):
    if not os.path.exists(fineDenoisedDIR):   # directory for fine denoised Raw micrographs.
        os.makedirs(fineDenoisedDIR)
        
    if skipStep7:
        fineModelPath = fineModel
    else:
        digits = int(np.ceil(np.log10(fineEpochs)))
        fineModelPath = fineSavePrefix + ('_epoch{:0'+str(digits)+'}.sav').format(fineEpochs)
    
    patchSizeParam = "" if patchSize == -1 else f"-s {patchSize}"
    status8 = os.system(f'python {NT2CDIR}denoiser/denoise_cmd.py -c 512 -o {fineDenoisedDIR} -m {fineModelPath} -d {cudaDevice} {patchSizeParam} {rawDataDIR}*.mrc')

    if status8 != 0:
        print("----Step 8 was not successfully finished with error code {}".format(status8))
        quit()

def randomExtract():
    if not os.path.exists(noisePatchDIR):       # directory for extracted noise patches.
        os.makedirs(noisePatchDIR)
    if extractDraw:
        if not os.path.exists(noiseDrawDIR):    # directory for visualization of extraction.
            os.makedirs(noiseDrawDIR)

    noiseDrawOption = '' if extractDraw == False else f'--noiseDraw {noiseDrawDIR}'
    statusRE = os.system(f'python {NT2CDIR}script/randomExtract.py --denoised {coarseDenoisedDIR} --raw {rawDataDIR} --noisePatch {noisePatchDIR} {noiseDrawOption} --worker {extractCore} -s 512 -n {randomPatchNum} 2> {workspaceDIR}noiseExtractSTDERR.log')
    if statusRE != 0:
        print("----Random Extract was not successfully finished with error code {}".format(statusRE))
        quit()

def gaussApplication():
    if not os.path.exists(noisyDIR):       # directory for gaussian noisy patches.
        os.makedirs(noisyDIR)

    noiseDrawOption = '' if extractDraw == False else f'--noiseDraw {noiseDrawDIR}'
    statusGA = os.system(f'python {NT2CDIR}workUtil/gaussApplier.py -i {cleanDIR} -o {noisyDIR} -a 1 --std {stdMultGauss}')
    if statusGA != 0:
        print("----Gauss Application was not successfully finished with error code {}".format(statusGA))
        quit()

def summaryWriter(step1, step2, step3, step4, step5, step6, step7, step8):
    with open(f"{workspaceDIR}summary.txt", "w") as resultLOG:
        resultLOG.write("WORKFLOW : {}\n".format(workflow))
        resultLOG.write("---------- SUMMARY 1 : time statistics ---------------------------------------------------\n")
        resultLOG.write("Elapsed time for step1 : {}\n".format(step2 - step1))
        resultLOG.write("Elapsed time for step2 : {}\n".format(step3 - step2))
        resultLOG.write("Elapsed time for step3 : {}\n".format(step4 - step3))
        resultLOG.write("Elapsed time for step4 : {}\n".format(step5 - step4))
        resultLOG.write("Elapsed time for step5 : {}\n".format(step6 - step5))
        if step7:
            resultLOG.write("Elapsed time for step6 : {}\n".format(step7 - step6))
        else:
            resultLOG.write("Elapsed time for step6 : {}\n".format(time.time() - step6))

        if step8:
            resultLOG.write("Elapsed time for step7 : {}\n".format(step8 - step7))
            resultLOG.write("Elapsed time for step8 : {}\n".format(time.time() - step8))
        resultLOG.write("Total Elapsed time     : {}\n".format(time.time() - pipelineStartTime))
        resultLOG.write("\n\n")
        resultLOG.write("---------- SUMMARY 2 : parameters statistics ---------------------------------------------\n")
        resultLOG.write("BASIC PARAMETERS ==========================\n")
        resultLOG.write("used NT2CDIR           : {}\n".format(NT2CDIR))
        resultLOG.write("workspaceDIR           : {}\n".format(workspaceDIR))
        resultLOG.write("cleanDataDIR           : {}\n".format(cleanDIR))
        resultLOG.write("noisyDataDIR           : {}\n".format(noisyDIR))
        resultLOG.write("cudaDevice             : {}\n".format(cudaDevice))
        resultLOG.write("coarseEpochs           : {}\n".format(coarseEpochs))
        resultLOG.write("rawDataDIR             : {}\n".format(rawDataDIR))
        resultLOG.write("extractCore            : {}\n".format(extractCore))
        resultLOG.write("generatorIter          : {}\n".format(generatorIter))
        resultLOG.write("synNum64               : {}\n".format(synNum64))
        resultLOG.write("augNum                 : {}\n".format(augNum))
        resultLOG.write("fineEpochs             : {}\n".format(fineEpochs))
        resultLOG.write("fineBatch              : {}\n".format(fineBatch))
        resultLOG.write("fineLR                 : {}\n".format(fineLR))
        resultLOG.write("\n\n")
        resultLOG.write("BYPASS PARAMETERS ==========================\n")
        resultLOG.write("coarseModel            : {}\n".format(coarseModel))
        resultLOG.write("coarseDenoised         : {}\n".format(coarseDenoised))
        resultLOG.write("noisePatch             : {}\n".format(noisePatch))
        resultLOG.write("loadGenerator          : {}\n".format(loadGenerator))
        resultLOG.write("synNoise               : {}\n".format(synNoise))
        resultLOG.write("fragmentNoisy          : {}\n".format(fragmentNoisy))
        resultLOG.write("fragmentClean          : {}\n".format(fragmentClean))
        resultLOG.write("noiseReweight          : {}\n".format(noiseReweight))
        resultLOG.write("bulkRenamed            : {}\n".format(bulkRenamed))
        resultLOG.write("fineModel              : {}\n".format(fineModel))
        resultLOG.write("\n\n")
        resultLOG.write("DEBUGGING PARAMETERS ==========================\n")
        resultLOG.write("extractDraw            : {}\n".format(extractDraw))
        resultLOG.write("synGrid                : {}\n".format(synGrid))
        resultLOG.write("trainGrid              : {}\n".format(trainGrid))
        resultLOG.write("\n\n")

def standardWorkflow():
    skipStep1           = False if (coarseModel == '') and (coarseDenoised == '') and (noisePatch == '') and (fineModel == '') else True
    skipStep2           = False if (coarseDenoised == '') and (noisePatch == '') and (fineModel == '') else True
    skipStep3           = False if (noisePatch == '') and (synNoise == '') and (fineModel == '') else True
    skipStep4           = False if (loadGenerator == '') and (synNoise == '') and (fineModel == '') else True
    skipStep5           = False if (synNoise == '') and (fineModel == '') else True
    skipStep6           = False if (fineModel == '') else True
    skipStep7           = False if (fineModel == '') else True
    # 1. Coarse Denoiser Training. ##########################################################################################################
    print("--Step 1 : Coarse Denoiser Training! -----------------------------------------------")
    step1StartTime      = time.time()
    if skipStep1:
        print("---- Step 1 skipped.")
    else:
        step1CoarseTrain()
        print("----Elapsed time for step 1 : {} seconds".format(time.time() - step1StartTime))
    # 2. Coarse Denoise Raw Micrographs. ####################################################################################################
    print("--Step 2 : Coarse Denoising! -------------------------------------------------------")
    step2StartTime      = time.time()
    if skipStep2:
        print("---- Step 2 skipped.")
    else:
        step2CoraseDenoise(skipStep1)
        print("----Elapsed time for step 2 : {} seconds".format(time.time() - step2StartTime))
    # 3. Noise extract. #####################################################################################################################
    print("--Step 3 : Noise extraction! -------------------------------------------------------")
    step3StartTime      = time.time()
    if skipStep3:
        print("---- Step 3 skipped.")
    else:
        step3NoiseExtract()
        print("----Elapsed time for step 3 : {} seconds".format(time.time() - step3StartTime))
    if untilExtract:
        quit()
    # 4. GAN noise synthesizer training. ####################################################################################################
    print("--Step 4 : GAN noise synthesizer training! -----------------------------------------")
    step4StartTime      = time.time()
    if skipStep4:
        print("---- Step 4 skipped.")
    else:
        step4GANtrain()
        print("----Elapsed time for step 4 : {} seconds".format(time.time() - step4StartTime))

    # 5. GAN noise synthesize. ##############################################################################################################
    print("--Step 5 : Noise synthesize! -------------------------------------------------------")
    step5StartTime      = time.time()
    if skipStep5:
        print("---- Step 5 skipped.")
    else:
        step5GANsynthesize()
        print("----Elapsed time for step 5 : {} seconds".format(time.time() - step5StartTime))
    # 6. Fragment then Noise reweighting. ###################################################################################################
    print("--Step 6 : Fragment then Noise reweighting! -----------------------------------------")
    step6StartTime      = time.time()

    if skipStep6:
        print('---- Step 6 skipped.')
    else:
        step6FragmentReweight()
        print("----Elapsed time for step 6 : {} seconds".format(time.time() - step6StartTime))
    # 7. Fine Denoiser training. ############################################################################################
    print("--Step 7 : Fine Denoiser training! -----------------------------------------")
    step7StartTime      = time.time()

    if skipStep7:
        print('---- Step 7 skipped.')
    else:
        retVal = step7Finetrain()
        print("----Elapsed time for step 7 : {} seconds".format(time.time() - step7StartTime))
    
    
    step8StartTime = time.time()
    if skipFineDenoise:
        print('---- Step 8 skipped.')
    elif retVal == 0:
        # 8. Fine denoise raw micrographs. ######################################################################################################
        print("--Step 8 : Fine denoise raw micrographs! -----------------------------------------")
        step8FineDenoise(skipStep7)
        print("----Elapsed time for step 8 : {} seconds".format(time.time() - step8StartTime))
    else:
        print("--Step 8 was skipped due to the error during the step 7.")
    
    summaryWriter(step1StartTime, step2StartTime, step3StartTime, step4StartTime, step5StartTime, step6StartTime, step7StartTime, step8StartTime)
    print(f"---- Result Log is saved to {workspaceDIR}summary.txt.")

def randomWorkflow(withGaussain=False):
    skipStep1           = False if (noisePatch == '') and (synNoise == '') and (fineModel == '') else True
    skipStep2           = False if (loadGenerator == '') and (synNoise == '') and (fineModel == '') else True
    skipStep3           = False if (synNoise == '') and (fineModel == '') else True
    skipStep4           = False if (fineModel == '') else True
    skipStep5           = False if (fineModel == '') else True
    # 1. Coarse Denoiser Training. ##########################################################################################################
    print("--Step 1 : Random extraction! -------------------------------------------------------")
    step1StartTime      = time.time()
    if skipStep1:
        print("---- Step 1 skipped.")
    else:
        randomExtract()
        print("----Elapsed time for step 1 : {} seconds".format(time.time() - step1StartTime))
    
    # 2. GAN noise synthesizer training. ####################################################################################################
    print("--Step 2 : GAN noise synthesizer training! -----------------------------------------")
    step2StartTime      = time.time()
    if skipStep2:
        print("---- Step 2 skipped.")
    else:
        step4GANtrain()
        print("----Elapsed time for step 2 : {} seconds".format(time.time() - step2StartTime))

    # 3. GAN noise synthesize. ##############################################################################################################
    print("--Step 3 : Noise synthesize! -------------------------------------------------------")
    step3StartTime      = time.time()
    if skipStep3:
        print("---- Step 3 skipped.")
    else:
        step5GANsynthesize()
        print("----Elapsed time for step 3 : {} seconds".format(time.time() - step3StartTime))
    
    # 4. Fragment then Noise reweighting. ###################################################################################################
    print("--Step 4 : Fragment then Noise reweighting! -----------------------------------------")
    step4StartTime      = time.time()
    if skipStep4:
        print('---- Step 4 skipped.')
    else:
        if withGaussain and (not skipGauss):
            gaussApplication()
        step6FragmentReweight()
        print("----Elapsed time for step 4 : {} seconds".format(time.time() - step4StartTime))
    # 5. Bulk rename and Fine Denoiser training. ############################################################################################
    print("--Step 5 : Fine Denoiser training! -----------------------------------------")
    step5StartTime      = time.time()
    if skipStep5:
        print('---- Step 5 skipped.')
    else:
        retVal = step7Finetrain()
        print("----Elapsed time for step 5 : {} seconds".format(time.time() - step5StartTime))
    
    
    step6StartTime      = time.time()
    if skipFineDenoise:
        print('---- Step 6 skipped.')
    elif retVal == 0:
        # 6. Fine denoise raw micrographs. ######################################################################################################
        print("--Step 6 : Fine denoise raw micrographs! -----------------------------------------")
        step8FineDenoise(skipStep5)
        print("----Elapsed time for step 6 : {} seconds".format(time.time() - step6StartTime))
    else:
        print("--Step 6 was skipped due to the error during the step 5.")
    
    summaryWriter(step1StartTime, step2StartTime, step3StartTime, step4StartTime, step5StartTime, step6StartTime, None, None)
    print(f"---- Result Log is saved to {workspaceDIR}summary.txt.")

if __name__ == '__main__':
    # 1. Argument Parsing.
    args = parse_args_standard()
    pipelineStartTime   = time.time()

    workflow            = args.workflow
    
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
    synNoise            = args.synNoise
    synNum64            = args.synNum64
    synGrid             = args.synGrid
    trainGrid           = args.trainGrid
    fragmentClean       = args.fragmentClean
    fragmentNoisy       = args.fragmentNoisy
    noiseReweight       = args.noiseReweight
    augNum              = args.augNum
    fineModel           = args.fineModel
    fineEpochs          = args.fineEpochs
    fineBatch           = args.fineBatch
    fineLR              = args.fineLR
    patchSize           = args.patchSize

    skipGauss           = args.skipGauss
    randomPatchNum      = args.randomPatchNum
    stdMultGauss        = args.stdMultGauss
    untilExtract        = args.untilExtract

    skipFineDenoise     = args.skipFineDenoise
    # 2. Directory, path.
    coarseSavePrefix    = workspaceDIR + 'coarseModel/' + 'model'
    fineSavePrefix      = workspaceDIR + 'fineModel/' + 'model'
    
    coarseModelDIR      = workspaceDIR + 'coarseModel/'
    noiseDrawDIR        = workspaceDIR + 'noiseDraw/'
    
    coarseDenoisedDIR   = (workspaceDIR + 'coarseDenoised/')         if coarseDenoised == '' else coarseDenoised
    noisePatchDIR       = (workspaceDIR + 'noisePatch/')             if noisePatch == ''     else noisePatch
    generatorPath       = (workspaceDIR + 'generator.pkl')           if loadGenerator == ''  else loadGenerator
    fragmentNoisyDIR    = (workspaceDIR + 'fragmentNoisy/')          if fragmentNoisy == ''  else fragmentNoisy
    fragmentCleanDIR    = (workspaceDIR + 'fragmentClean/')          if fragmentClean == ''  else fragmentClean
    noiseReweightDIR    = (workspaceDIR + 'noiseReweight/')          if noiseReweight == ''  else noiseReweight
    synNoiseDIR         = (workspaceDIR + 'synthesized_noises/')     if synNoise == ''       else synNoise
    fineModelDIR        = (workspaceDIR + 'fineModel/')
    fineDenoisedDIR     = (workspaceDIR + 'fineDenoised/')

    if workflow == 'standard':
        standardWorkflow()
    elif workflow == 'random':
        randomWorkflow()
    elif workflow == 'randgauss':
        randomWorkflow(withGaussain = True)
    else:
        print("ERROR : There is no {} workflow".format(workflow))