from utils.config import parse_args
from utils.data_loader_512 import get_data_loader
from models.wgan_gradient_penalty_savemrc_512 import WGAN_GP

def main(args):
    model = WGAN_GP(args)

    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)

    # Start model training
    if args.is_train == 'True':
        model.train(train_loader, args.synGrid)

    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)
        # for i in range(50):
        #    model.generate_latent_walk(i)

if __name__ == '__main__':
    args = parse_args()
    main(args)