import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from utils.datasets_custom import CustomDataset

def get_data_loader(args):
    trans = transforms.Compose([
        transforms.ToPILImage(),            
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
    train_dataset = CustomDataset(root=args.dataroot, train=True, transform=trans)
    test_dataset  = CustomDataset(root=args.dataroot, train=False, transform=trans)

    # Check if everything is ok with loading datasets
    assert train_dataset
    assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader
