from utils.config import parse_args
from models.wgan_gradient_penalty_savemrc_320 import WGAN_GP

def main(args):
    model =  WGAN_GP(args)
    for i in range(1000):
        model.synthesize_noise(1 + 64 * i, args.load_D, args.load_G)

if __name__ == '__main__':
    args = parse_args()
    print("ARGS CUDA : ", args.cuda)
    main(args)

# CUDA_VISIBLE_DEVICES=5 python main_generateV_256.py --dataset custom --batch_size 64 --dataroot / --cuda True --load_D /cdata/db1/logs/discriminator.pkl --load_G /cdata/db1/logs/generator.pkl