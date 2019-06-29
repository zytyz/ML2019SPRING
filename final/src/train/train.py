# coding=utf-8
import argparse
import os
import urllib.request
import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop, RandomCrop
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from data import DatasetFromFolder
from model.rpnet import Net
#from model.attennet import AttnNet
from utils import save_checkpoint, del_checkpoint
import pytorch_ssim
from lossVgg import customLoss
import glob

# Training settings
parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--tag", type=str, help="tag for this training")
parser.add_argument("--rb", type=int, default=18, help="number of residual blocks")
parser.add_argument("--train", default="../datasets/IndoorTrain/", type=str,
                    help="path to load train datasets(default: none)")
parser.add_argument("--test", default="../datasets/IndoorTest/", type=str,
                    help="path to load test datasets(default: none)")
parser.add_argument("--batchSize", type=int, default=64, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=300, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=1e-4")
parser.add_argument("--reg", type=float, default=0.001, help="Regularization term for ssim loss. Default=1e-3")
parser.add_argument("--step", type=int, default=2000, help="step to test the model performance. Default=2000")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--gpus", type=int, default=4, help="nums of gpu to use")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--report", default=False, type=bool, help="report to wechat")
parser.add_argument("--loss", default='ssim', type=str, help="loss function: ssim or vggloss or MSE")


def main():
    global opt, name, logger, model, criterion ,criterion2
    opt = parser.parse_args()
    print(opt)

    # Tag_ResidualBlocks_BatchSize
    name = "%s_%d_%d" % (opt.tag, opt.rb, opt.batchSize)

    with open('logs/'+name + 'log.csv','a') as file:
        file.write(opt.loss+', \n')
        file.write('Epochs, Test Loss, Test PSNR, Test SSIM, Test score\n')


    logger = SummaryWriter("runs/" + name)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    seed = 1334
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    cudnn.benchmark = True

    print("==========> Loading datasets")

    train_dataset = DatasetFromFolder(opt.train, transform=Compose([
        ToTensor()
    ]), phase='train')

    indoor_test_dataset = DatasetFromFolder(opt.test, transform=Compose([
        ToTensor()
    ]), phase='test')

    training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=True)
    indoor_test_loader = DataLoader(dataset=indoor_test_dataset, num_workers=opt.threads, batch_size=1, pin_memory=True,
                                    shuffle=True)

    print("==========> Building model")
    model = Net(opt.rb)
    #model.half()
    criterion = nn.MSELoss(size_average=True)
    if opt.loss == 'ssim':
        criterion2 = pytorch_ssim.SSIM()
    elif opt.loss == 'vggloss':
        criterion2 = customLoss()
    elif opt.loss != 'MSE':
        raise RunTimeError('no loss')

    print(model)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] // 2 + 1
            model.load_state_dict(checkpoint["state_dict"])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['state_dict'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("==========> Setting GPU")
    if cuda:
        model = nn.DataParallel(model, device_ids=[i for i in range(opt.gpus)]).cuda()
        criterion = criterion.cuda()
        try:
            criterion2 = criterion2.cuda()
        except:
            print('loss is MSE, no criterion2')
    else:
        model = model.cpu()
        criterion = criterion.cpu()
        try:
            criterion2 = criterion2.cpu()
        except:
            print('loss is MSE, no criterion2')

    print("==========> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, eps=1e-4)

    best_score = 0
    print("==========> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, indoor_test_loader, optimizer, epoch)
        score = test(indoor_test_loader, epoch)
        #score = testLargeImage(opt.test)
        if score < best_score:
            best_score = score
            try:
                del_checkpoint(name)
            except:
                print('Nothing is deleted!')
            save_checkpoint(model, epoch, name)


def train(training_data_loader, indoor_test_loader, optimizer, epoch):
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])

    for iteration, batch in enumerate(training_data_loader, 1):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        steps = len(training_data_loader) * (epoch-1) + iteration

        data, label = \
            Variable(batch[0]), \
            Variable(batch[1], requires_grad=False)

        if opt.cuda:
            data = data.cuda()
            label = label.cuda()
        else:
            data = data.cpu()
            label = label.cpu()

        output = model(data)

        # loss = criterion(output, label) / (data.size()[0]*2)
        mse_loss = criterion(output, label)
        if opt.loss == 'MSE':
            loss = mse_loss

        elif opt.loss == 'ssim':
            ssim_loss = criterion2(output, label)
            #loss = (-1)* 10 * torch.log10(1.0 / mse_loss.data) * ssim_loss
            loss = mse_loss - opt.reg*ssim_loss

        elif opt.loss == 'vggloss':
            vgg_loss = criterion2(output, label)
            loss = mse_loss + 0.5 * vgg_loss

        loss.backward()

        # torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
        optimizer.step()

        if iteration % 10 == 0:
            if opt.loss == 'MSE':
                print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader),
                                                                    loss.data))
            elif opt.loss == 'ssim':
                print("===> Epoch[{}]({}/{}): Loss: {:.6f}, MSELoss: {:.6f}, SSIM: {:.6f}".format(epoch, iteration, len(training_data_loader),
                                                                    loss.data, mse_loss.data, ssim_loss.data))
            elif opt.loss == 'vggloss':
                print("===> Epoch[{}]({}/{}): Loss: {:.6f}, MSELoss: {:.6f}, VGGLoss: {:.6f}".format(epoch, iteration, len(training_data_loader),
                                                                    loss.data, mse_loss.data, vgg_loss.data))
            logger.add_scalar('loss', loss.data, steps)

        if iteration % opt.step == 0:
            data_temp = make_grid(data.data)
            label_temp = make_grid(label.data)
            output_temp = make_grid(output.data)

            logger.add_image('data_temp', data_temp, steps)
            logger.add_image('label_temp', label_temp, steps)
            logger.add_image('output_temp', output_temp, steps)


def test(test_data_loader, epoch):
    psnrs = []
    mses = []
    ssims = []
    for iteration, batch in enumerate(test_data_loader, 1):
        model.eval()
        data, label = \
            Variable(batch[0]), \
            Variable(batch[1])

        if opt.cuda:
            data = data.cuda()
            label = label.cuda()
        else:
            data = data.cpu()
            label = label.cpu()

        with torch.no_grad():
            output = model(data)
        output = torch.clamp(output, 0., 1.)
        mse = nn.MSELoss(size_average=True)(output, label).cpu()
        mses.append(mse.data)
        psnr = 10 * np.log10(1.0 / mse.data)
        psnrs.append(psnr)

        ssim_loss = pytorch_ssim.SSIM()
        ssim_out = (-1) * ssim_loss(output,label).cpu()
        ssims.append(ssim_out.data)



    psnr_mean = np.mean(psnrs)

    mse_mean = np.mean(mses)
    ssims_mean = np.mean(ssims)
    score = psnr_mean * ssims_mean

    print("Vaild  epoch %d psnr: %f ssim: %f score: %f" % (epoch, psnr_mean, ssims_mean, score))
    with open('logs/'+name + 'log.csv','a') as file:
        file.write(str(epoch)+','+str(psnr_mean)+','+str(ssims_mean)+','+str(score)+'\n')

    logger.add_scalar('psnr', psnr_mean, epoch)
    logger.add_scalar('mse', mse_mean, epoch)
    logger.add_scalar('score', score, epoch)

    data = make_grid(data.data)
    label = make_grid(label.data)
    output = make_grid(output.data)

    logger.add_image('data', data, epoch)
    logger.add_image('label', label, epoch)
    logger.add_image('output', output, epoch)

    if opt.report:
        urllib.request.urlopen(
            "https://sc.ftqq.com/SCU21303T3ae6f3b60b71841d0def9295e4a500905a7524916a85c.send?text=epoch_{}_loss_{}".format(
                epoch, psnr_mean))
    return score


if __name__ == "__main__":
    os.system('clear')
    main()