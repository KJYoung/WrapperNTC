import argparse
import os


def parse_args_standard():
    parser = argparse.ArgumentParser(description="Improved Wrapper of NoiseTransfer2Clean.")
    
    # Workflow selection
    parser.add_argument('-w', '--workflow', choices=['standard', 'random', 'randgauss'], default='standard', help='choose the workflow (default: standard).')

    # Basic Parameters
    parser.add_argument('--nt2cDir', required=True, help='path to nt2c.')
    parser.add_argument('--workspace', required=True, help='path to save all the outputs.')
    parser.add_argument('--noisy', required=True, help='path to noisy dataset.')
    parser.add_argument('--clean', required=True, help='path to clean dataset.')
    parser.add_argument('--raw', required=True, help='path to raw micrographs dataset.')
    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, set to -1 to force CPU (default: 0).')
    parser.add_argument('--coarseEpochs', default=100, type=int, help='coarse denoiser training epochs (default: 100).')
    parser.add_argument('--extractCores', default=2, type=int, help='number of CPU cores to use for noise extract (default: 2).')
    parser.add_argument('--generator_iters', type=int, default=10000, help='The number of iterations for generator in WGAN model (default: 10000)')
    parser.add_argument('--synNum64', type=int, default=320, help='number of noise synthesize. [synNum64 * 64] patches would be generated (default: 320).')
    parser.add_argument('--augNum', type=int, default=2, help='number of noise reweighting augmentation. [augNum] patches would be generated (default: 2).')
    parser.add_argument('--fineEpochs', default=400, type=int, help='fine denoiser training epochs (default: 400).')
    parser.add_argument('--fineBatch', default=4, type=int, help='fine denoiser training batch size (default: 4).')
    parser.add_argument('--fineLR', default=0.002, type=float, help='fine denoiser training learning rate (* default: 0.002 *).')
    
    # Bypass Options.
    parser.add_argument('--coarseModel', default='', type=str, help='already trained coarse denoiser model path to skip step 1.')
    parser.add_argument('--coarseDenoised', default='', type=str, help='already coarse denoised inputs path to skip step 1~2.')
    parser.add_argument('--noisePatch', default='', type=str, help='already extracted noise patches path to skip step 1~3.')
    parser.add_argument('--load_G', default='', type=str, help='already trained GAN noise synthesizer(generator) path to skip step 4.')
    parser.add_argument('--synNoise', default='', type=str, help='already synthesized noise patches path to skip step 4~5.')
    parser.add_argument('--fragmentNoisy', default='', type=str, help='already fragmented noise patches path to skip step 6 partially,')
    parser.add_argument('--fragmentClean', default='', type=str, help='already fragmented clean patches path to skip step 6 partially,')
    parser.add_argument('--noiseReweight', default='', type=str, help='already reweighted noisy patches path to skip step 6 partially.')
    parser.add_argument('--bulkRenamed', default='', type=str, help='already paired bulkRenamed clean patches path to skip step 7 partially.')
    parser.add_argument('--fineModel', default='', type=str, help='already trained fine denoiser path to skip step 1~7 entirely.')
    
    # Debugging Options.
    parser.add_argument('--extractDraw', dest='extractDraw', action='store_true', help='save noiseDraw during step 3 if this option is flagged.')
    parser.add_argument('--synGrid',     dest='synGrid',     action='store_true', help='save overview grid during step 5 if this option is flagged.')
    parser.add_argument('--trainGrid',   dest='trainGrid',   action='store_true', help='save overview grid during step 4 if this option is flagged.')
    return check_args(parser.parse_args())

# Checking arguments
def check_args(args):
    # --extractCores
    try:
        assert args.extractCores >= 1
    except:
        print('extractCores must be >= 1')
        quit()
    
    # --augNum
    try:
        assert args.extractCores >= 1
    except:
        print('augNum must be >= 1')
        quit()

    # --coarseEpochs
    try:
        assert args.coarseEpochs >= 10
    except:
        print('coarseEpochs must be >= 10')
        quit()

    # --fineEpochs
    try:
        assert args.fineEpochs >= 10
    except:
        print('fineEpochs must be >= 10')
        quit()
    
    # --fineBatch
    try:
        assert args.fineBatch >= 2
    except:
        print('fineBatch must be >= 10')
        quit()

    # --fineLR
    try:
        assert args.fineLR >= 0.0001
    except:
        print('fineLR must be >= 0.0001')
        quit()

    # --nt2c_dir
    if not os.path.exists(args.nt2cDir):
        print('NT2C directory does not exist!')
        quit()
    if not args.nt2cDir.endswith('/'):
        print('WARNING : nt2cDir option should end with "/"!')
        args.nt2cDir = args.nt2cDir + '/'
    
    # --workspace
    if not os.path.exists(args.workspace):
        print('Workspace directory does not exist!')
        quit()
    if not args.workspace.endswith('/'):
        print('WARNING : workspace option should end with "/"!')
        args.workspace = args.workspace + '/'

    # --clean
    if not os.path.exists(args.clean): 
        print('Clean data directory does not exist!')
        quit()
    if not args.clean.endswith('/'):
        print('WARNING : clean option should end with "/"!')
        args.clean = args.clean + '/'
    
    # --noisy
    if not os.path.exists(args.noisy):
        print('Noisy data directory does not exist!')
        quit()
    if not args.noisy.endswith('/'):
        print('WARNING : noisy option should end with "/"!')
        args.noisy = args.noisy + '/'

    # --raw
    if not os.path.exists(args.raw):
        print('Raw data directory does not exist!')
        quit()
    if not args.raw.endswith('/'):
        print('WARNING : raw option should end with "/"!')
        args.raw = args.raw + '/'
    
    # --coarseModel
    if not args.coarseModel == '':
        if not os.path.exists(args.coarseModel):
            print('coarseModel does not exist!')
            quit()
    
    # --coarseDenoised
    if not args.coarseDenoised == '':
        if not os.path.exists(args.coarseDenoised):
            print('coarseDenoised directory does not exist!')
            quit()
        if not args.coarseDenoised.endswith('/'):
            print('WARNING : coarseDenoised option should end with "/"!')
            args.coarseDenoised = args.coarseDenoised + '/'
    
    # --noisePatch
    if not args.noisePatch == '':
        if not os.path.exists(args.noisePatch):
            print('noisePatch directory does not exist!')
            quit()
        if not args.noisePatch.endswith('/'):
            print('WARNING : noisePatch option should end with "/"!')
            args.noisePatch = args.noisePatch + '/'
    
    # --synNoise
    if not args.synNoise == '':
        if not os.path.exists(args.synNoise):
            print('synNoise directory does not exist!')
            quit()
        if not args.synNoise.endswith('/'):
            print('WARNING : synNoise option should end with "/"!')
            args.synNoise = args.synNoise + '/'

    # --fragmentNoisy
    if not args.fragmentNoisy == '':
        if not os.path.exists(args.fragmentNoisy):
            print('fragmentNoisy directory does not exist!')
            quit()
        if not args.fragmentNoisy.endswith('/'):
            print('WARNING : fragmentNoisy option should end with "/"!')
            args.fragmentNoisy = args.fragmentNoisy + '/'
    # --fragmentClean
    if not args.fragmentClean == '':
        if not os.path.exists(args.fragmentClean):
            print('fragmentClean directory does not exist!')
            quit()
        if not args.fragmentClean.endswith('/'):
            print('WARNING : fragmentClean option should end with "/"!')
            args.fragmentClean = args.fragmentClean + '/'

    # --noiseReweight
    if not args.noiseReweight == '':
        if not os.path.exists(args.noiseReweight):
            print('noiseReweight directory does not exist!')
            quit()
        if not args.noiseReweight.endswith('/'):
            print('WARNING : noiseReweight option should end with "/"!')
            args.noiseReweight = args.noiseReweight + '/'

    # --bulkRenamed
    if not args.bulkRenamed == '':
        if not os.path.exists(args.bulkRenamed):
            print('bulkRenamed directory does not exist!')
            quit()
        if not args.bulkRenamed.endswith('/'):
            print('WARNING : bulkRenamed option should end with "/"!')
            args.bulkRenamed = args.bulkRenamed + '/'
    
    # --fineModel
    if not args.fineModel == '':
        if not os.path.exists(args.fineModel):
            print('fineModel does not exist!')
            quit()
    
    args.channels = 1
    return args
