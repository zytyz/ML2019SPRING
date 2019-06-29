#!/usr/bin/env python
import argparse
import torch
import utils
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Resize

from model.rpnet import Net
import pytorch_ssim
import glob
import numpy as np
import os

parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--rb", type=int, default=13, help="number of residual blocks")
parser.add_argument("--checkpoint", type=str, help="path to load model checkpoint")
parser.add_argument("--test", type=str, help="path to load test images")
parser.add_argument("--testGT", type=str, help="path to load test GT")

opt = parser.parse_args()
print(opt)

net = Net(opt.rb)
net.load_state_dict(torch.load(opt.checkpoint)['state_dict'])
net.eval()
#net = nn.DataParallel(net, device_ids=[0, 1, 2, 3]).cuda()
net = net.cuda()

#print(net)

images = glob.glob(opt.test+'/*')
imagesGT = glob.glob(opt.testGT+'/*')
images.sort()
imagesGT.sort()

mse = nn.MSELoss(size_average=True)
mse = mse.cuda()
ssim = pytorch_ssim.SSIM()
ssim = ssim.cuda()

scores = []

for i in range(len(images)):
    #filename = im_path.split('/')[-1]
    #print(filename)
    im = Image.open(images[i])
    h, w = im.size
    print(h, w)
    im = ToTensor()(im)
    im = Variable(im).view(1, -1, w, h)
    im = im.cuda()
    with torch.no_grad():
        im = net(im)
    im = torch.clamp(im, 0., 1.)

    imGT = Image.open(imagesGT[i])
    imGT = ToTensor()(imGT).view(1, -1, w, h)
    imGT = imGT.cuda()

    mse_loss = mse(im,imGT)

    psnr = 10 * np.log10(1.0 / mse_loss.data.cpu())
    ssim_loss = ssim(im,imGT)
    score = psnr * ssim_loss.data.cpu()
    scores.append(score)
    print(score)

    im = im.cpu()
    im = im.data[0]
    im = ToPILImage()(im)
    im.save('output/%s' % os.path.basename(images[i]))

print(np.mean(scores))
