from utils.config import parse_args
from utils.data_loader_512 import get_data_loader
from models.wgan_gradient_practice import WGAN_GP

def main(args):
    model = WGAN_GP(args)
    train_loader, test_loader = get_data_loader(args)

    # Start model training
    if args.is_train == 'True':
        model.train(train_loader)

    else:
        model.evaluate(test_loader, args.load_D, args.load_G)

if __name__ == '__main__':
    args = parse_args()
    main(args)