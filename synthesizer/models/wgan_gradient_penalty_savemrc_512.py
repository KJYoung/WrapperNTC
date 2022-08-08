import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from itertools import chain
from torchvision import utils

import mrcfile

SAVE_PER_TIMES = 200

class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100 # torch.Size([64, 100, 1, 1])
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # torch.Size([64, 1024, 4, 4])
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # torch.Size([64, 512, 8, 8])
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # torch.Size([64, 256, 16, 16])
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            # # torch.Size([64, 128, 32, 32])
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            
            # # torch.Size([64, 64, 64, 64])
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            # # torch.Size([64, 64, 256, 256])
            nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # # torch.Size([64, 1, 512, 512])
            ########################################################################################################
        self.output = nn.Tanh()

    def forward(self, x):
        #print("Generator get shape", x.shape) # torch.Size([64, 100, 1, 1])
        x = self.main_module(x)
        #print(" After main_module shape4", x.shape) # torch.Size([64, 1, 128, 128])
        # print(" Generator output shape", (self.output(x)).shape) # torch.Size([64, 1, 128, 128])
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            
            # Image (Cx256x256)
            #nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=4, stride=2, padding=1),
            #nn.InstanceNorm2d(32, affine=True),
            #nn.LeakyReLU(0.2, inplace=True),

            #State (32*128*128)
            #nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            #State (64*64*64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),            
            
            #State (128*32*32)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class WGAN_GP(object):
    def __init__(self, args):
        # print("WGAN_GradientPenalty init model.")
        self.G = Generator(args.channels)
        self.D = Discriminator(args.channels)
        self.C = args.channels

        # Check if cuda is available
        self.check_cuda(args.cuda)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 16

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.number_of_images = 10

        self.generator_iters = args.generator_iters
        self.critic_iter = 5
        self.lambda_term = 10

        self.outputDir = args.output_dir

    def get_torch_variable(self, arg):
        if self.cuda:
            #print("cuda index: {}".format(self.cuda_index))
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        # print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            # print("cuda index: {}".format(self.cuda_index))
            # print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def train(self, train_loader, synGrid):
        self.t_begin = t.time()

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True
            d_loss=0
            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                images = self.data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue

                z = torch.rand((self.batch_size, 100, 1, 1))

                images, z = self.get_torch_variable(images), self.get_torch_variable(z)

                # Train discriminator : WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))

                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                #gradient_penalty.backward()
                #edit by lihongjia 
                gradient_penalty.backward(retain_graph=True)

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            # train generator : compute loss with fake images
            z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
            
            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % SAVE_PER_TIMES == 0:
                self.save_model(g_iter)

                if not os.path.exists(self.outputDir + 'training_result_images/'):
                    os.makedirs(self.outputDir + 'training_result_images/')

                # Denormalize images and save them in grid 8x8
                # z = self.get_torch_variable(torch.randn(800, 100, 1, 1))
                z = self.get_torch_variable(torch.randn(64, 100, 1, 1))
                samples = self.G(z)
                samples = samples.mul(0.5).add(0.5)
                samples = samples.data.cpu()

                if synGrid:
                    grid = utils.make_grid(samples, nrow=8)
                    utils.save_image(grid, self.outputDir + 'training_result_images/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))
                
                with mrcfile.new(self.outputDir + 'training_result_images/img_generatori_iter_{}.mrc'.format(str(g_iter)),overwrite=True) as n_out:
                    n_out.set_data(samples.data.cpu().numpy()[0])
                
                # Testing
                time = t.time() - self.t_begin
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time))

        print('Time of training-{}'.format((t.time() - self.t_begin)))
        # Save the trained parameters
        self.save_model(g_iter)

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        print(samples.shape, type(samples))
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def synthesize_noise(self, loopNum, synGrid, G_model_path):
        if not os.path.exists(self.outputDir + 'synthesized_noises/'): # directory for mrc files.
            os.makedirs(self.outputDir + 'synthesized_noises/')
        if synGrid and (not os.path.exists(self.outputDir + 'synthesized_grid/')): # directory for overview grid images.
            os.makedirs(self.outputDir + 'synthesized_grid/')
        self.load_generator(G_model_path)

        for i in range(loopNum):
            startNum = 1 + 64*i
            z = self.get_torch_variable(torch.randn(64, 100, 1, 1))
            samples = self.G(z)
            samples = samples.mul(0.5).add(0.5)
            samples = samples.data.cpu()
            samplesNP = samples.numpy()
            for sampleNum in range(64):
                noiseID = startNum + sampleNum
                noise_out=mrcfile.new(self.outputDir + 'synthesized_noises/synthesized_noise_{}.mrc'.format(str(noiseID)), overwrite=True)
                noise_out.set_data(samplesNP[sampleNum])

            if synGrid:
                grid = utils.make_grid(samples)
                gridName = self.outputDir + 'synthesized_grid/overview_synthesized_{}-{}.png'.format(str(startNum), str(startNum+64-1))
                utils.save_image(grid, gridName)
            if i % 20 == 0:
                print("Synthesize {} done.".format(i))
    
    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 128, 128)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 128, 128)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 128, 128))
            else:
                generated_images.append(sample.reshape(128, 128))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self, iterNum):
        torch.save(self.G.state_dict(), self.outputDir + 'generator.pkl')
        torch.save(self.D.state_dict(), self.outputDir + 'discriminator.pkl')

        if iterNum % 5000 == 0 and iterNum != 0:
            torch.save(self.G.state_dict(), self.outputDir + f'generator_{iterNum}.pkl')
            torch.save(self.D.state_dict(), self.outputDir + f'discriminator_{iterNum}.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def load_generator(self, G_model_filename):
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")
