import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

#model 54 code:
class MyCnn(nn.Module):
    def __init__(self):
        super(MyCnn, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1,64,4,2,1), # [64, 24, 24]
            nn.BatchNorm2d(64)
        )
        self.cnn = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [64, 12, 12]
            nn.Dropout(p=0.2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [128, 6, 6]
            nn.Dropout(p=0.2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),       # [256, 3, 3]
            nn.Dropout(p=0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 7)
        )
        self.cnn.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        out = self.cnn(self.cnn1(x))
        out = out.view(out.size()[0], -1)
        return self.fc(out)

    def getlayeroutput(self,x):
        return self.cnn1(x)

parser = argparse.ArgumentParser()
parser.add_argument('-num','--modelnum',type=int)
parser.add_argument('-path',type=str)
args = parser.parse_args()
print(args)

train_X = np.load('torch_train_X.npy').reshape(-1,1,48,48)
train_Y = np.load('torch_train_Y.npy').reshape(-1,1)

with open('ind_list_'+str(args.modelnum)+'.txt') as f:
    ind_list = f.read().split(' ')
    ind_list = [int(x) for x in ind_list]
print('ind list {}'.format(ind_list[:10]))
val_len = int( 1/4 * train_Y.shape[0])

val_X = train_X[ind_list[:val_len]]
val_Y = train_Y[ind_list[:val_len]]
train_X = train_X[ind_list[val_len:]]
train_Y = train_Y[ind_list[val_len:]]

print('train X shape {}'.format(train_X.shape))
print('train Y shape {}'.format(train_Y.shape))
print('val X shape {}'.format(val_X.shape))
print('val Y shape {}'.format(val_Y.shape))


device = torch.device('cuda')
print(device)

_, express_ind = np.unique(val_Y,return_index=True)
express_ind = np.ndarray.tolist(express_ind)
print('express_ind: {}'.format(express_ind))

x_data = torch.tensor(val_X[express_ind]).float().to(device) 
y_data = torch.tensor(val_Y[express_ind,0]).to(device)
print('x data shape {}'.format(x_data.shape))
print('y data shape {}'.format(y_data.shape))
print(y_data)


model = torch.load('best_model_'+str(args.modelnum)+'.pkl')
model.to(device)

def compute_saliency_maps(x, y, model):
    model.eval()
    x.requires_grad_()
    y_pred = model(x.cuda())
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()
    saliency = x.grad.abs().squeeze().data
    return saliency

def show_saliency_maps(x, y, model):
    x_org = x.squeeze().detach().cpu().numpy()
    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(x, y, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.detach().cpu().numpy()
    
    num_pics = x_org.shape[0]
    for i in range(num_pics):
        # You need to save as the correct fig names
        #plt.imsave('saliency/pic_'+ str(i), x_org[i], cmap=plt.cm.gray)
        plt.imsave(args.path + 'fig1_'+ str(i) +'.jpg', saliency[i], cmap=plt.cm.jet)

# using the first ten images for example
show_saliency_maps(x_data, y_data, model)
'''saliency = compute_saliency_maps(x_data, y_data, model)
print('x data shape {}'.format(x_data.shape))
print('y data shape {}'.format(y_data.shape))
print('saliency shape {}'.format(saliency.shape))
print(saliency[0])'''








