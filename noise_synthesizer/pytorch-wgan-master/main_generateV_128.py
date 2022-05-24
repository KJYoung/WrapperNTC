from utils.config import parse_args
from utils.data_loader_128 import get_data_loader

from models.gan import GAN
from models.dcgan import DCGAN_MODEL
from models.wgan_clipping import WGAN_CP
from models.wgan_gradient_penalty_savemrc_128 import WGAN_GP

def main(args):
    model =  WGAN_GP(args)
    for i in range(10):
        model.synthesize_noise(1 + 64 * i, args.load_D, args.load_G)

if __name__ == '__main__':
    args = parse_args()
    print("ARGS CUDA : ", args.cuda)
    main(args)
