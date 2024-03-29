from __future__ import print_function,division

import sys
import numpy as np

import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from utils.data.loader import load_image

def load_model(name):

    if name == 'none':
        return Identity()

    # set the name aliases
    if name == 'unet':
        name = 'unet_L2_v0.2.2.sav'
    elif name == 'xunet':
        name = 'xunet.sav'

    # construct model and load the state
    if name == 'unet_L2_v0.2.2.sav':
        model = UDenoiseNet(base_width=11, top_width=5)
    # elif name == 'unet_L2_v0.2.1.sav':
    #     model = UDenoiseNet(base_width=7, top_width=3)
    elif name == 'xunet':
        model = None
    else:
        # if not set to a pretrained model, try loading path directly
        return torch.load(name)

    # load the pretrained model parameters
    import pkg_resources
    pkg = __name__
    path = 'pretrained/denoise/' + name
    f = pkg_resources.resource_stream(pkg, path)
    state_dict = torch.load(f) # load the parameters

    model.load_state_dict(state_dict)

    return model


def denoise(model, x, patch_size=-1, padding=128):
    # check the patch plus padding size
    # print(patch_size, padding)

    use_patch = False
    if patch_size > 0:
        s = patch_size + padding
        use_patch = (s < x.size(0)) or (s < x.size(1))

    if use_patch:
        return denoise_patches(model, x, patch_size, padding=padding)

    torch.cuda.empty_cache()
    with torch.no_grad():
        x = x.unsqueeze(0).unsqueeze(0)
        t = model(x)
        y = t.squeeze()

    return y


def denoise_patches(model, x, patch_size, padding=128):
    y = torch.zeros_like(x)
    x = x.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        for i in range(0, x.size(2), patch_size):
            for j in range(0, x.size(3), patch_size):
                # include padding extra pixels on either side
                si = max(0, i - padding)
                ei = min(x.size(2), i + patch_size + padding)

                sj = max(0, j - padding)
                ej = min(x.size(3), j + patch_size + padding)

                xij = x[:,:,si:ei,sj:ej]
                yij = model(xij).squeeze() # denoise the patch

                # match back without the padding
                si = i - si
                sj = j - sj

                y[i:i+patch_size,j:j+patch_size] = yij[si:si+patch_size,sj:sj+patch_size]

    return y

class UDenoiseNet(nn.Module):
    # U-net from noise2noise paper
    def __init__(self, nf=48, base_width=11, top_width=3):
        super(UDenoiseNet, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv2d(1, nf, base_width, padding=base_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec5 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(2*nf+1, 64, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, top_width, padding=top_width//2)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, x], 1)

        y = self.dec1(h)

        return y

class PairedImages:
    def __init__(self, x, y, crop=800, xform=True, preload=False, cutoff=0):
        self.x = x
        self.y = y
        self.crop = crop
        self.xform = xform
        self.cutoff = cutoff

        self.preload = preload
        if preload:
            self.x = [self.load_image(p) for p in x]
            self.y = [self.load_image(p) for p in y]

    def load_image(self, path):
        x = np.array(load_image(path), copy=False)
        x = x.astype(np.float32) # make sure dtype is single precision
        mu = x.mean()
        std = x.std()
        if std == 0.0:
            x = x - mu
        else:
            x = (x - mu)/std
        if self.cutoff > 0:
            x[(x < -self.cutoff) | (x > self.cutoff)] = 0
        return x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        if self.preload:
            x = self.x[i]
            y = self.y[i]
        else:
            x = self.load_image(self.x[i])
            y = self.load_image(self.y[i])

        # randomly crop
        if self.crop is not None:
            size = self.crop

            n,m = x.shape
            i = np.random.randint(n-size+1)
            j = np.random.randint(m-size+1)

            x = x[i:i+size, j:j+size]
            y = y[i:i+size, j:j+size]

        # randomly flip
        if self.xform:
            if np.random.rand() > 0.5:
                x = np.flip(x, 0)
                y = np.flip(y, 0)
            if np.random.rand() > 0.5:
                x = np.flip(x, 1)
                y = np.flip(y, 1)


            k = np.random.randint(4)
            x = np.rot90(x, k=k)
            y = np.rot90(y, k=k)

            # swap x and y
            if np.random.rand() > 0.5:
                t = x
                x = y
                y = t

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)

        return x, y


class NoiseImages:
    def __init__(self, x, crop=800, xform=True, preload=False, cutoff=0):
        self.x = x
        self.crop = crop
        self.xform = xform
        self.cutoff = cutoff

        self.preload = preload
        if preload:
            x = [self.load_image(p) for p in x]

    def load_image(self, path):
        x = np.array(load_image(path), copy=False)
        mu = x.mean()
        std = x.std()
        if std == 0.0:
            x = x - mu
        else:
            x = (x - mu)/std
        if self.cutoff > 0:
            x[(x < -self.cutoff) | (x > self.cutoff)] = 0
        return x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        if self.preload:
            x = self.x[i]
        else:
            x = self.load_image(self.x[i])

        # randomly crop
        if self.crop is not None:
            size = self.crop

            n,m = x.shape
            i = np.random.randint(n-size+1)
            j = np.random.randint(m-size+1)

            x = x[i:i+size, j:j+size]

        # randomly flip
        if self.xform:
            if np.random.rand() > 0.5:
                x = np.flip(x, 0)
            if np.random.rand() > 0.5:
                x = np.flip(x, 1)

            k = np.random.randint(4)
            x = np.rot90(x, k=k)

        x = np.ascontiguousarray(x)

        return x

class L0Loss:
    def __init__(self, eps=1e-8, gamma=2):
        self.eps = eps
        self.gamma = gamma

    def __call__(self, x, y):
        return torch.mean((torch.abs(x - y) + self.eps)**self.gamma)


def eval_noise2noise(model, dataset, criteria, batch_size=10
                    , use_cuda=False, num_workers=0):
    data_iterator = torch.utils.data.DataLoader(dataset, batch_size=batch_size
                                               , num_workers=num_workers)

    n = 0
    loss = 0

    model.eval()
        
    with torch.no_grad():
        for x1,x2 in data_iterator:
            if use_cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()

            x1 = x1.unsqueeze(1)
            y = model(x1).squeeze(1)

            loss_ = criteria(y, x2).item()

            b = x1.size(0)
            n += b
            delta = b*(loss_ - loss)
            loss += delta/n

    return loss


def train_noise2noise(model, dataset, lr=0.001, optim='adagrad', batch_size=10, num_epochs=100
                     , criteria=nn.MSELoss(), dataset_val=None
                     , use_cuda=False, num_workers=0, shuffle=True):

    gamma = None
    if criteria == 'L0':
        gamma = 2
        eps = 1e-8
        criteria = L0Loss(eps=eps, gamma=gamma)
    elif criteria == 'L1':
        criteria = nn.L1Loss()
    elif criteria == 'L2':
        criteria = nn.MSELoss()
    
    if optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == 'adagrad':
        optim = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)
    data_iterator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle
                                               , num_workers=num_workers)

    total = len(dataset)

    for epoch in range(1, num_epochs+1):
        model.train()
        
        n = 0
        loss_accum = 0

        if gamma is not None:
            # anneal gamma to 0
            criteria.gamma = 2 - (epoch-1)*2/num_epochs

        for x1,x2 in data_iterator:
            if use_cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()

            x1 = x1.unsqueeze(1)
            y = model(x1).squeeze(1)

            loss = criteria(y, x2)
            
            loss.backward()
            optim.step()
            optim.zero_grad()

            loss = loss.item()
            b = x1.size(0)

            n += b
            delta = b*(loss - loss_accum)
            loss_accum += delta/n

            print('# [{}/{}] {:.2%} loss={:.5f}'.format(epoch, num_epochs, n/total, loss_accum)
                 , file=sys.stderr, end='\r')
        print(' '*80, file=sys.stderr, end='\r')

        if dataset_val is not None:
            loss_val = eval_noise2noise(model, dataset_val, criteria
                                       , batch_size=batch_size
                                       , num_workers=num_workers
                                       , use_cuda=use_cuda
                                       )
            yield epoch, loss_accum, loss_val
        else:
            yield epoch, loss_accum