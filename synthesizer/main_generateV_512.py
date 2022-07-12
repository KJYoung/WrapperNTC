from utils.config import parse_args
from models.wgan_gradient_penalty_savemrc_512 import WGAN_GP

def main(args):
    model =  WGAN_GP(args)
    model.synthesize_noise(320, args.load_G)

if __name__ == '__main__':
    args = parse_args()
    print("ARGS CUDA : ", args.cuda)
    main(args)

# CUDA_VISIBLE_DEVICES=7 python /cdata/NT2C/noise_synthesizer/pytorch-wgan-master/main_generateV_512.py --dataset custom --batch_size 64 --dataroot / --cuda True --load_D /cdata/db1/size512/discriminator.pkl --load_G /cdata/db1/size512/generator.pkl
# ==> load_D deleted.
# CUDA_VISIBLE_DEVICES=7 python /cdata/NT2C/noise_synthesizer/pytorch-wgan-master/main_generateV_512.py --dataset custom --batch_size 64 --dataroot / --cuda True --load_G /cdata/db1/size512/generator.pkl