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

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="day2night", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=1024, help="size of image height")
parser.add_argument("--img_width", type=int, default=1024, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--levels", type=int, default=4, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=3, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=5.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=1.0, help="identity loss weight")
parser.add_argument("--lambda_adv", type=float, default=3.0, help="identity loss weight")
parser.add_argument('--gpu_id', default='0', type=str, help='learning rate for the stochastic gradient update.')
parser.add_argument('--validate_size', default=5, type=int, help='number of images to validate.')
opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

# Create sample and checkpoint directories
os.makedirs("../images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("../saved_models/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()
if_conv = True

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)
Trans_AB = Trans_high()
Trans_BA = Trans_high()

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    Trans_AB = Trans_AB.cuda()
    Trans_BA = Trans_BA.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("../saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch - 1)))
    G_BA.load_state_dict(torch.load("../saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch - 1)))
    D_A.load_state_dict(torch.load("../saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch - 1)))
    D_B.load_state_dict(torch.load("../saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch - 1)))
    Trans_AB.load_state_dict(torch.load("../saved_models/%s/Trans_AB_%d.pth" % (opt.dataset_name, opt.epoch - 1)))
    Trans_BA.load_state_dict(torch.load("../saved_models/%s/Trans_BA_%d.pth" % (opt.dataset_name, opt.epoch - 1)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)
    Trans_AB.apply(weights_init_normal)
    Trans_BA.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters(), Trans_AB.parameters(), Trans_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
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
    mask_0, mask_1, mask_2, mask_3 = trans_net(high_with_low, if_conv)
    pyr_result[-2] = torch.mul(pyr_original[-2], mask_0) #+ pyr_original[-2]
    pyr_result[-3] = torch.mul(pyr_original[-3], mask_1) #+ pyr_original[-3]
    pyr_result[-4] = torch.mul(pyr_original[-4], mask_2) #+ pyr_original[-4]
    pyr_result[-5] = torch.mul(pyr_original[-5], mask_3) #+ pyr_original[-5]

    return pyr_result

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    Trans_AB.eval()
    Trans_BA.eval()

    real_A_full = Variable(imgs["A"].type(Tensor))
    pyr_A  = pyramid.pyramid_decom(img=real_A_full, max_levels=opt.levels)
    real_A = pyr_A[-1]
    fake_B = G_AB(real_A)
    pyr_A_trans = pyramid_transform(pyr_A, real_A, fake_B, Trans_AB, if_conv)
    fake_B_full = pyramid.pyramid_recons(pyr_A_trans)

    real_B_full = Variable(imgs["B"].type(Tensor))
    start = time.time()
    pyr_B  = pyramid.pyramid_decom(img=real_B_full, max_levels=opt.levels)
    real_B = pyr_B[-1]
    fake_A = G_BA(real_B)
    pyr_B_trans = pyramid_transform(pyr_B, real_B, fake_A, Trans_BA, if_conv)
    fake_A_full = pyramid.pyramid_recons(pyr_B_trans)

    cost = (time.time() - start) / opt.validate_size
    print('time cost for one image: {:.4f}'.format(cost))

    # Arange images along x-axis
    real_A = make_grid(torch.clamp(real_A_full, -1, 1).cpu().data, nrow=5, normalize=True)
    real_B = make_grid(torch.clamp(real_B_full, -1, 1).cpu().data, nrow=5, normalize=True)
    fake_A = make_grid(torch.clamp(fake_A_full, -1, 1).cpu().data, nrow=5, normalize=True)
    fake_B = make_grid(torch.clamp(fake_B_full, -1, 1).cpu().data, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
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
        # save_image(real_A, "1.png", normalize=False)
        # save_image(image_recons, "2.png", normalize=False)
        # print(image_recons.shape)

        # print(real_A.shape)

        real_A = pyr_A[-1]
        real_B = pyr_B[-1]

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()
        Trans_AB.train()
        Trans_BA.train()

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        pyr_A_trans = pyramid_transform(pyr_A, real_A, fake_B, Trans_AB, if_conv)
        fake_B_full = pyramid.pyramid_recons(pyr_A_trans)
        loss_GAN_AB = criterion_GAN(D_B(fake_B_full), valid)

        fake_A = G_BA(real_B)
        pyr_B_trans = pyramid_transform(pyr_B, real_B, fake_A, Trans_BA, if_conv)
        fake_A_full = pyramid.pyramid_recons(pyr_B_trans)
        loss_GAN_BA = criterion_GAN(D_A(fake_A_full), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        pyr_A_recons  = pyramid.pyramid_decom(img=fake_B_full, max_levels=opt.levels)
        recov_A_low = G_BA(pyr_A_recons[-1])
        pyr_A_recons_trans = pyramid_transform(pyr_A_recons, pyr_A_recons[-1], recov_A_low, Trans_BA, if_conv)
        recov_A_full = pyramid.pyramid_recons(pyr_A_recons_trans)
        loss_cycle_A = criterion_cycle(recov_A_full, real_A_full)

        pyr_B_recons  = pyramid.pyramid_decom(img=fake_A_full, max_levels=opt.levels)
        recov_B_low = G_AB(pyr_B_recons[-1])
        pyr_B_recons_trans = pyramid_transform(pyr_B_recons, pyr_B_recons[-1], recov_B_low, Trans_AB, if_conv)
        recov_B_full = pyramid.pyramid_recons(pyr_B)
        loss_cycle_B = criterion_cycle(recov_B_full, real_B_full)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = opt.lambda_adv * loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A_full), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A_full)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B_full), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B_full)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

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
                loss_cycle.item(),
                loss_identity.item(),
                time.time() - start_time,
            )
        )

        # If at sample interval save image
        if opt.sample_interval != -1 and batches_done % opt.sample_interval == 0:
           sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "../saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
        torch.save(G_BA.state_dict(), "../saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), "../saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), "../saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
        torch.save(Trans_AB.state_dict(), "../saved_models/%s/Trans_AB_%d.pth" % (opt.dataset_name, epoch))
        torch.save(Trans_BA.state_dict(), "../saved_models/%s/Trans_BA_%d.pth" % (opt.dataset_name, epoch))
