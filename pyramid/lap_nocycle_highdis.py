import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *
import pyramid_torch as pyramid

import cv2

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="day2night", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=20, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=1024, help="size of image height")
parser.add_argument("--img_width", type=int, default=1024, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--levels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=4, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=1.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=0.0, help="identity loss weight")
parser.add_argument("--lambda_adv", type=float, default=1.0, help="identity loss weight")
parser.add_argument('--gpu_id', default='0', type=str, help='learning rate for the stochastic gradient update.')
parser.add_argument('--validate_size', default=5, type=int, help='number of images to validate.')
opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)



cuda = torch.cuda.is_available()
if_conv = True

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G = GeneratorResNet(input_shape, opt.n_residual_blocks)
D = Discriminator(input_shape)
Trans = Trans_high()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss()

if cuda:
    G = G.cuda()
    D = D.cuda()
    Trans = Trans.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G.load_state_dict(torch.load("../saved_models/%s/G_%d.pth" % (opt.dataset_name, opt.epoch - 1)))
    D.load_state_dict(torch.load("../saved_models/%s/D_%d.pth" % (opt.dataset_name, opt.epoch - 1)))
    Trans.load_state_dict(torch.load("../saved_models/%s/Trans_%d.pth" % (opt.dataset_name, opt.epoch - 1)))
else:
    # Initialize weights
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)
    Trans.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_Trans = torch.optim.Adam(Trans.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_Trans = torch.optim.lr_scheduler.LambdaLR(
    optimizer_Trans, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

transforms_val = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset("../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    ImageDataset("../data/%s" % opt.dataset_name, transforms_=transforms_val, unaligned=True, mode="test"),
    batch_size=opt.validate_size,
    shuffle=True,
    num_workers=10,
)

def pyramid_transform(pyr_original, real_low, fake_low, trans_net, if_conv):
    pyr_result = pyr_original
    pyr_result[-1] = fake_low
    real_B_up = nn.functional.upsample_bilinear(real_low, scale_factor=2)
    fake_A_up = nn.functional.upsample_bilinear(fake_low, scale_factor=2)
    high_with_low = torch.cat([pyr_original[-2], real_B_up], 1)
    high_with_low = torch.cat([high_with_low, fake_A_up], 1)
    mask_0, mask_1, mask_2 = trans_net(high_with_low, if_conv)
    pyr_result[-2] = torch.mul(pyr_original[-2], mask_0) + pyr_original[-2]
    pyr_result[-3] = torch.mul(pyr_original[-3], mask_1) + pyr_original[-3]
    pyr_result[-4] = torch.mul(pyr_original[-4], mask_2) + pyr_original[-4]
    # pyr_result[-5] = torch.mul(pyr_original[-5], mask_3) + pyr_original[-5]
    
    return pyr_result

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G.eval()
    Trans.eval()

    real_full = Variable(imgs["A"].type(Tensor))
    start = time.time()
    pyr  = pyramid.pyramid_decom(img=real_full, max_levels=opt.levels)
    fake = G(pyr[-1])
    pyr_trans = pyramid_transform(pyr, pyr[-1], fake, Trans, if_conv)
    fake_full = pyramid.pyramid_recons(pyr_trans)
    cost = (time.time() - start) / opt.validate_size
    print('time cost for one image: {:.4f}'.format(cost))

    # Arange images along x-axis
    real_A = make_grid(torch.clamp(real_full, -1, 1).cpu().data, nrow=opt.validate_size, normalize=True)
    fake_B = make_grid(torch.clamp(fake_full, -1, 1).cpu().data, nrow=opt.validate_size, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B), 1)
    if not os.path.exists("../images/%s" % (opt.dataset_name)):
        os.makedirs("../images/%s" % (opt.dataset_name))
    save_image(image_grid, "../images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        start_time = time.time()

        # Set model input
        real_A_full = Variable(batch["A"].type(Tensor))
        real_B_full = Variable(batch["B"].type(Tensor))

        pyr_A  = pyramid.pyramid_decom(img=real_A_full, max_levels=opt.levels)
        pyr_B  = pyramid.pyramid_decom(img=real_B_full, max_levels=opt.levels)

        real_A = pyr_A[-1]
        real_B = pyr_B[-1]

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A_full.size(0), *D.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A_full.size(0), *D.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G.train()
        optimizer_G.zero_grad()
        
        # Identity loss
        loss_identity = criterion_identity(G(real_A), real_A)

        # GAN loss
        fake_B = G(real_A)
        pyr_A_trans = pyramid_transform(pyr_A, real_A, fake_B, Trans, if_conv)
        fake_B_full = pyramid.pyramid_recons(pyr_A_trans)
        # fake_B_full = torch.clamp(fake_B_full, -1, 1)

        
        loss_GAN = criterion_GAN(D(fake_B_full), valid)
        loss_recons = criterion_cycle(fake_B_full, real_A_full)
        loss_G = opt.lambda_adv * loss_GAN + opt.lambda_cyc * loss_recons

        loss_G.backward(retain_graph=True)
        optimizer_G.step()


        Trans.train()
        optimizer_Trans.zero_grad()
        loss_Trans = opt.lambda_cyc * loss_recons + opt.lambda_adv * loss_GAN
        loss_Trans.backward()
        optimizer_Trans.step()

        # -----------------------
        #  Train Discriminator
        # -----------------------

        if i % 1 == 0:
            optimizer_D.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D(real_B_full), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B_full)
            loss_fake = criterion_GAN(D(fake_B_.detach()), fake)
            # Total loss

            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        # batches_left = opt.n_epochs * len(dataloader) - batches_done
        # time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        # prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch {}/{}] [Batch {}/{}] [D loss: {:.4f}] [G loss: {:.4f}, adv: {:.4f}, cycle: {:.4f}, identity: {:.4f}, cost_time: {:.4f}]".format(
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_recons.item(),
                loss_identity.item(),
                time.time() - start_time,
            )
        )

        # If at sample interval save image
        if opt.sample_interval != -1 and batches_done % opt.sample_interval == 0:
           sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    lr_scheduler_Trans.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        if not os.path.exists("../saved_models/%s" % (opt.dataset_name)):
            os.makedirs("../saved_models/%s" % (opt.dataset_name))
        torch.save(G.state_dict(), "../saved_models/%s/G_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D.state_dict(), "../saved_models/%s/D_%d.pth" % (opt.dataset_name, epoch))
        torch.save(Trans.state_dict(), "../saved_models/%s/Trans_%d.pth" % (opt.dataset_name, epoch))
