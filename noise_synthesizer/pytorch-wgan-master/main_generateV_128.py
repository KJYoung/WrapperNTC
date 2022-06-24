from utils.config import parse_args
from models.wgan_gradient_penalty_savemrc_128 import WGAN_GP

def main(args):
    model =  WGAN_GP(args)
    for i in range(10):
        model.synthesize_noise(1 + 64 * i, args.load_D, args.load_G)

if __name__ == '__main__':
    args = parse_args()
    print("ARGS CUDA : ", args.cuda)
    main(args)

# python3 main_generateV_128.py --dataset custom --batch_size 64 --dataroot /cdata/NT2C/denoiser/noiseExtract/np --cuda True --load_D ./discriminator.pkl --load_G ./generator.pkl