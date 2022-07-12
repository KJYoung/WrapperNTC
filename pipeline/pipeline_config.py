import argparse
import os


def parse_args_standard():
    parser = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")
    parser.add_argument('--nt2cDir', required=True, help='path to nt2c.')
    parser.add_argument('--workspace', required=True, help='path to save all the outputs.')
    parser.add_argument('--clean', required=True, help='path to clean dataset.')
    parser.add_argument('--noisy', required=True, help='path to noisy dataset.')
    parser.add_argument('--raw', required=True, help='path to raw micrographs dataset.')
    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, set to -1 to force CPU (default: 0)')
    parser.add_argument('--coarseEpochs', default=200, type=int, help='coarse denoiser training epochs.')
    parser.add_argument('--extractCores', default=2, type=int, help='number of CPU cores to use for noise extract')
    parser.add_argument('--generator_iters', type=int, default=10000, help='The number of iterations for generator in WGAN model.')
    
    # Bypass Options.
    parser.add_argument('--coarseModel', default='', type=str, help='already trained coarse denoiser model path to skip step 1.')
    parser.add_argument('--coarseDenoised', default='', type=str, help='already coarse denoised inputs path to skip step 1~2.')
    parser.add_argument('--noisePatch', default='', type=str, help='already extracted noise patches path to skip step 1~3.')
    parser.add_argument('--load_G', default='', type=str, , help='already trained GAN noise synthesizer(generator) path to skip step 4.')

    # Debugging Options.
    parser.add_argument('--extractDraw', default=False, type=bool, help='save noiseDraw during step 3 if this option is true.')

    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--load_D', type=str, default='False', help='Path for loading Discriminator network')
    parser.add_argument('--load_G', type=str, default='False', help='Path for loading Generator network')
    return check_args(parser.parse_args())

# Checking arguments
def check_args(args):
    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one')

    try:
        assert args.extractCores >= 2
    except:
        print('Number of noise extract core must be larger than or equal to two')
    # --nt2c_dir
    if not os.path.exists(args.nt2cDir):
        print('NT2C directory does not exist!')
        assert False
    if not args.nt2cDir.endswith('/'):
        print('WARNING : nt2cDir option should end with "/"!')
        args.nt2cDir = args.nt2Dir + '/'
    
    # --workspace
    if not os.path.exists(args.workspace):
        print('Workspace directory does not exist!')
        assert False
    if not args.workspace.endswith('/'):
        print('WARNING : workspace option should end with "/"!')
        args.workspace = args.workspace + '/'

    # --clean
    if not os.path.exists(args.clean): 
        print('Clean data directory does not exist!')
        assert False
    if not args.clean.endswith('/'):
        print('WARNING : clean option should end with "/"!')
        args.clean = args.clean + '/'
    
    # --noisy
    if not os.path.exists(args.noisy):
        print('Noisy data directory does not exist!')
        assert False
    if not args.noisy.endswith('/'):
        print('WARNING : noisy option should end with "/"!')
        args.noisy = args.noisy + '/'

    # --raw
    if not os.path.exists(args.raw):
        print('Raw data directory does not exist!')
        assert False
    if not args.raw.endswith('/'):
        print('WARNING : raw option should end with "/"!')
        args.raw = args.raw + '/'
    
    # --coarseModel
    if not args.coarseModel == '':
        if not os.path.exists(args.coarseModel):
            print('coarseModel does not exist!')
            assert False
    
    # --coarseDenoised
    if not args.coarseDenoised == '':
        if not os.path.exists(args.coarseDenoised):
            print('coarseDenoised directory does not exist!')
            assert False
        if not args.coarseDenoised.endswith('/'):
            print('WARNING : coarseDenoised option should end with "/"!')
            args.coarseDenoised = args.coarseDenoised + '/'
    
    # --noisePatch
    if not args.noisePatch == '':
        if not os.path.exists(args.noisePatch):
            print('noisePatch directory does not exist!')
            assert False
        if not args.noisePatch.endswith('/'):
            print('WARNING : noisePatch option should end with "/"!')
            args.noisePatch = args.noisePatch + '/'
    
    args.channels = 1
    return args
