import os
import numpy as np
import torch
from torch.optim import Adam
from torchvision import models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam,SGD
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

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

def save_image(path,data):
    data = data.reshape(48,48)
    plt.imsave(path, data, cmap=plt.cm.gray)

class CNNLayerVisualization():
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        #if not os.path.exists('filter_image'):
            #os.makedirs('filter_image')

    def visual_layer(self):
        image = np.zeros((1,1,48,48))
        image = torch.from_numpy(image).float().to(device)
        image = Variable(image, requires_grad=True)

        optimizer = Adam([image], lr=0.1)

        for i in range(1, 1001):
            optimizer.zero_grad()

            x = image
            #print('update {} , x: {}'.format(i,x))
            #print(x)
            #print('layer input x shape: {}'.format(x.shape))

            x = self.model.cnn1(x)
            self.conv_output = x[0, self.selected_filter]
            #print(self.conv_output.shape)
            #print(self.conv_output)

            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = (-1) * torch.mean(self.conv_output)
            #if i%100==0:
                #print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            #print(image)
            
            # Save image
        #im_path = 'filter_image/filter_'+str(self.selected_filter)+'.png'
        #save_image(im_path, image.detach().cpu().numpy())
        return image.detach().cpu().numpy()


parser = argparse.ArgumentParser()
parser.add_argument('-num','--modelnum',type=int)
parser.add_argument('-filter_num',type=int,default=64)
parser.add_argument('-path',type=str)
args = parser.parse_args()
print(args)
device = torch.device('cuda')
print(device)

cnn_layer = 0
filter_pos = 14
# Fully connected layer is not needed

model = MyCnn() #model 54 code
model = torch.load('best_model_'+str(args.modelnum)+'.pkl')
model.to(device)

images = []

for filter_pos in range(args.filter_num):
    print('filter_pos {}'.format(filter_pos))
    layer_vis = CNNLayerVisualization(model, cnn_layer, filter_pos)
    # Layer visualization with pytorch hooks
    img = layer_vis.visual_layer()
    images.append(img)

for it in range(100//10):
    fig = plt.figure(figsize = (14, 8))
    for i in range(args.filter_num):
        q = fig.add_subplot(args.filter_num/16, 16, i + 1)
        #raw = images[i][it][0].squeeze()
        q.imshow(images[i].reshape(48,48), cmap = 'Oranges')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        #plt.xlabel('{:.3f}'.format(filter_img[i][it][1]))
        plt.tight_layout()
    fig.savefig(args.path + 'fig2_1.jpg')


train_X = np.load('torch_train_X.npy').reshape(-1,1,48,48)
train_Y = np.load('torch_train_Y.npy').reshape(-1,1)

print('train X shape {}'.format(train_X.shape))
print('train Y shape {}'.format(train_Y.shape))


image = train_X[0].reshape(1,1,48,48)
image = torch.from_numpy(image).float().to(device)
image = Variable(image, requires_grad=True)

model = MyCnn() #model 54 code
model = torch.load('best_model_'+str(args.modelnum)+'.pkl')
model.to(device)

x = model.cnn1(image)

#if not os.path.exists('filter_image_2'):
    #os.makedirs('filter_image_2')

images = []

for filter_pos in range(args.filter_num):
    
    tmp = x[0, filter_pos].detach().cpu().numpy()

    #im_path = 'filter_image_2/filter_'+str(filter_pos)+'.png'
    #save_image(im_path, tmp)
    images.append(tmp)


for it in range(100//10):
    fig = plt.figure(figsize = (14, 8))
    for i in range(args.filter_num):
        q = fig.add_subplot(args.filter_num/16, 16, i + 1)
        #raw = images[i][it][0].squeeze()
        q.imshow(images[i].reshape(24,24), cmap = 'Oranges')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        #plt.xlabel('{:.3f}'.format(filter_img[i][it][1]))
        plt.tight_layout()
    fig.savefig(args.path + 'fig2_2.jpg')