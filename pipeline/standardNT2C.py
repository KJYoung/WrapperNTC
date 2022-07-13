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
    synNoise            = args.synNoise
    synNum64            = args.synNum64
    synGrid             = args.synGrid
    trainGrid           = args.trainGrid
    fragmentClean       = args.fragmentClean
    fragmentNoisy       = args.fragmentNoisy
    noiseReweight       = args.noiseReweight
    augNum              = args.augNum
    bulkRenamed         = args.bulkRenamed
    fineModel           = args.fineModel
    fineEpochs          = args.fineEpochs
    fineBatch           = args.fineBatch
    fineLR              = args.fineLR

    coarseSavePrefix    = workspaceDIR + 'coarseModel/' + 'model'
    coarseModelDIR      = workspaceDIR + 'coarseModel/'
    coarseDenoisedDIR   = workspaceDIR + 'coarseDenoised/'
    noisePatchDIR       = workspaceDIR + 'noisePatch/'
    noiseDrawDIR        = workspaceDIR + 'noiseDraw/'
    generatorPath       = (workspaceDIR + 'generator.pkl')          if loadGenerator == ''  else loadGenerator
    fragmentNoisyDIR    = (workspaceDIR + 'fragmentNoisy/')         if fragmentNoisy == ''  else fragmentNoisy
    fragmentCleanDIR    = (workspaceDIR + 'fragmentClean/')         if fragmentClean == ''  else fragmentClean
    noiseReweightDIR    = (workspaceDIR + 'noiseReweight/')          if noiseReweight == ''  else noiseReweight
    synNoiseDIR         = (workspaceDIR + 'synthesized_noises/')     if synNoise == ''       else synNoise
    bulkRenamedDIR      = (workspaceDIR + 'fragmentPairedClean/')   if bulkRenamed == ''    else bulkRenamed
    fineSavePrefix      = workspaceDIR + 'fineModel/' + 'model'
    fineModelDIR        = (workspaceDIR + 'fineModel/')
    fineDenoisedDIR     = (workspaceDIR + 'fineDenoised/')

    skipStep1           = False if (coarseModel == '') and (coarseDenoised == '') and (noisePatch == '') and (fineModel == '') else True
    skipStep2           = False if (coarseDenoised == '') and (noisePatch == '') and (fineModel == '') else True
    skipStep3           = False if (noisePatch == '') and (fineModel == '') else True
    skipStep4           = False if (loadGenerator == '') and (synNoise == '') and (fineModel == '') else True
    skipStep5           = False if (synNoise == '') and (fineModel == '') else True
    skipStep6           = False if (fineModel == '') else True
    skipStep7           = False if (fineModel == '') else True

    # 1. Coarse Denoiser Training. #####################################################################################################################
    print("--Step 1 : Coarse Denoiser Training! -----------------------------------------------")
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
    # 2. Coarse Denoise Raw Micrographs. #####################################################################################################################
    print("--Step 2 : Coarse Denoising! -------------------------------------------------------")
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
    # 3. Noise extract. #####################################################################################################################
    print("--Step 3 : Noise extraction! -------------------------------------------------------")
    step3StartTime      = time.time()
    if skipStep3:
        print("---- Step 3 skipped.")
    else:
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
    # 4. GAN noise synthesizer training. #####################################################################################################################
    print("--Step 4 : GAN noise synthesizer training! -----------------------------------------")
    step4StartTime      = time.time()
    if skipStep4:
        print("---- Step 4 skipped.")
    else:
        GANtrainDataDIR = (workspaceDIR + "noisePatch/") if noisePatch == "" else noisePatch

        if cudaDevice >= 0:
            if trainGrid:
                status4 = os.system(f'CUDA_VISIBLE_DEVICES={cudaDevice} python {NT2CDIR}synthesizer/main_savemrc_512.py --batch_size 16 --dataroot {GANtrainDataDIR} --cuda True --generator_iters {generatorIter} --output_dir {workspaceDIR} --synGrid > {workspaceDIR}GANtrain{generatorIter}.log 2> {workspaceDIR}GANtrainSTDERR.log')
            else:
                status4 = os.system(f'CUDA_VISIBLE_DEVICES={cudaDevice} python {NT2CDIR}synthesizer/main_savemrc_512.py --batch_size 16 --dataroot {GANtrainDataDIR} --cuda True --generator_iters {generatorIter} --output_dir {workspaceDIR} > {workspaceDIR}GANtrain{generatorIter}.log 2> {workspaceDIR}GANtrainSTDERR.log')
        else:
            if trainGrid:
                status4 = os.system(f'python {NT2CDIR}synthesizer/main_savemrc_512.py --batch_size 16 --dataroot {GANtrainDataDIR} --cuda False --generator_iters {generatorIter} --output_dir {workspaceDIR} --synGrid > {workspaceDIR}GANtrain{generatorIter}.log 2> {workspaceDIR}GANtrainSTDERR.log')
            else:
                status4 = os.system(f'python {NT2CDIR}synthesizer/main_savemrc_512.py --batch_size 16 --dataroot {GANtrainDataDIR} --cuda False --generator_iters {generatorIter} --output_dir {workspaceDIR} > {workspaceDIR}GANtrain{generatorIter}.log 2> {workspaceDIR}GANtrainSTDERR.log')
        if status4 != 0:
            print("----Step 4 was not successfully finished with error code {}".format(status4))
            quit()
        print("----Elapsed time for step 4 : {} seconds".format(time.time() - step4StartTime))

    # 5. GAN noise synthesize. #####################################################################################################################
    print("--Step 5 : Noise synthesize! -------------------------------------------------------")
    step5StartTime      = time.time()
    if skipStep5:
        print("---- Step 5 skipped.")
    else:
        if cudaDevice >= 0:
            if synGrid:
                status5 = os.system(f'CUDA_VISIBLE_DEVICES={cudaDevice} python {NT2CDIR}synthesizer/main_generateV_512.py --batch_size 64 --dataroot / --cuda True --output_dir {workspaceDIR} --load_G {generatorPath} --synNum64 {synNum64} --synGrid 2> /dev/null')
            else:
                status5 = os.system(f'CUDA_VISIBLE_DEVICES={cudaDevice} python {NT2CDIR}synthesizer/main_generateV_512.py --batch_size 64 --dataroot / --cuda True --output_dir {workspaceDIR} --load_G {generatorPath} --synNum64 {synNum64} 2> /dev/null')
        else:
            if synGrid:
                status5 = os.system(f'python {NT2CDIR}synthesizer/main_generateV_512.py --batch_size 64 --dataroot / --cuda False --output_dir {workspaceDIR} --load_G {generatorPath} --synNum64 {synNum64} --synGrid 2> /dev/null')
            else:
                status5 = os.system(f'python {NT2CDIR}synthesizer/main_generateV_512.py --batch_size 64 --dataroot / --cuda False --output_dir {workspaceDIR} --load_G {generatorPath} --synNum64 {synNum64} 2> /dev/null')
        if status5 != 0:
            print("----Step 5 was not successfully finished with error code {}".format(status5))
            quit()
        print("----Elapsed time for step 5 : {} seconds".format(time.time() - step5StartTime))
    # 6. Fragment then Noise reweighting. #####################################################################################################################
    print("--Step 6 : Fragment then Noise reweighting! -----------------------------------------")
    step6StartTime      = time.time()

    if skipStep6:
        print('---- Step 6 skipped.')
    else:
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
            status6_3 = os.system(f'python {NT2CDIR}script/add_norm_noise_512.py {fragmentCleanDIR} {synNoiseDIR} {fragmentNoisyDIR} {noiseReweightDIR} {augNum}')
            if status6_3 != 0:
                print("----Step 6-3 : Noise reweighting was not successfully finished with error code {}".format(status6_3))
                quit()
        
        print("----Elapsed time for step 6 : {} seconds".format(time.time() - step6StartTime))
    # 7. Bulk rename and Fine Denoiser training. #####################################################################################################################
    print("--Step 7 : Bulk rename and Fine Denoiser training! -----------------------------------------")
    step7StartTime      = time.time()

    if skipStep7:
        print('---- Step 7 skipped.')
    else:
        if bulkRenamed == '':
            if not os.path.exists(bulkRenamedDIR):
                os.makedirs(bulkRenamedDIR)
            status7_1 = os.system(f'python {NT2CDIR}workUtil/bulkRenamer.py {fragmentCleanDIR} {bulkRenamedDIR} {augNum}')
            if status7_1 != 0:
                print("----Step 7-1 : Bulk renaming was not successfully finished with error code {}".format(status7_1))
                quit()

        if not os.path.exists(fineModelDIR):      # directory for fine Denoiser models.
            os.makedirs(fineModelDIR)
        status7_2 = os.system(f'python {NT2CDIR}denoiser/denoise_cmd.py -a {noiseReweightDIR} -b {bulkRenamedDIR} -d {cudaDevice} -c 512 --num-epochs {fineEpochs} --lr {fineLR} --batch-size {fineBatch} --save-prefix {fineSavePrefix} > {workspaceDIR}fineSTDOUT.log')
        if status7_2 != 0:
            print("----Step 7-2 : Fine denoiser training was not successfully finished with error code {}".format(status7_2))
            quit()
        print("----Elapsed time for step 7 : {} seconds".format(time.time() - step7StartTime))
    # 8. Fine denoise raw micrographs. #####################################################################################################################
    print("--Step 8 : Fine denoise raw micrographs! -----------------------------------------")
    step8StartTime      = time.time()
    if not os.path.exists(fineDenoisedDIR):   # directory for fine denoised Raw micrographs.
        os.makedirs(fineDenoisedDIR)
        
    if skipStep7:
        fineModelPath = fineModel
    else:
        digits = int(np.ceil(np.log10(fineEpochs)))
        fineModelPath = fineSavePrefix + ('_epoch{:0'+str(digits)+'}.sav').format(fineEpochs)
    status8 = os.system(f'python {NT2CDIR}denoiser/denoise_cmd.py -c 512 -o {fineDenoisedDIR} -m {fineModelPath} -d {cudaDevice} {rawDataDIR}*.mrc')
    if status8 != 0:
        print("----Step 8 was not successfully finished with error code {}".format(status8))
        quit()

    print("----Elapsed time for step 8 : {} seconds".format(time.time() - step8StartTime))
    
    with open(f"{workspaceDIR}summary.txt", "w") as resultLOG:
        resultLOG.write("---------- SUMMARY 1 : time statistics ---------------------------------------------------\n")
        resultLOG.write("Elapsed time for step1 : {}\n".format(step2StartTime - step1StartTime))
        resultLOG.write("Elapsed time for step2 : {}\n".format(step3StartTime - step2StartTime))
        resultLOG.write("Elapsed time for step3 : {}\n".format(step4StartTime - step3StartTime))
        resultLOG.write("Elapsed time for step4 : {}\n".format(step5StartTime - step4StartTime))
        resultLOG.write("Elapsed time for step5 : {}\n".format(step6StartTime - step5StartTime))
        resultLOG.write("Elapsed time for step6 : {}\n".format(step7StartTime - step6StartTime))
        resultLOG.write("Elapsed time for step7 : {}\n".format(step8StartTime - step7StartTime))
        resultLOG.write("Elapsed time for step8 : {}\n".format(time.time()    - step8StartTime))
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
    print(f"---- Result Log is saved to {workspaceDIR}summary.txt.")

if __name__ == '__main__':
    args = parse_args_standard()
    main(args)