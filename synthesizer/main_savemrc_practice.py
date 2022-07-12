from utils.config import parse_args
from utils.data_loader_320 import get_data_loader

from models.gan import GAN
from models.dcgan import DCGAN_MODEL
from models.wgan_clipping import WGAN_CP
#from models.wgan_gradient_penalty import WGAN_GP
from models.wgan_gradient_practice import WGAN_GP

def main(args):
    model = WGAN_GP(args)
    
    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)

    # Start model training
    if args.is_train == 'True':
        model.train(train_loader)

    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)
        # for i in range(50):
        #    model.generate_latent_walk(i)

if __name__ == '__main__':
    args = parse_args()
    print(args.cuda)
    main(args)
