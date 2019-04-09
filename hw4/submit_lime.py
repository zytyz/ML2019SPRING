import argparse
import numpy as np
import torch
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import os,sys
import lime
from lime import lime_image

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
# load data and model
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

'''_, express_ind = np.unique(val_Y,return_index=True)
express_ind = np.ndarray.tolist(express_ind)
print('express_ind: {}'.format(express_ind))
express_ind.pop(2)
express_ind.append(4)'''

express_ind = [9,70,26,27,44,23,10]


device = torch.device('cuda')
print(device)

model = torch.load('best_model_'+str(args.modelnum)+'.pkl')
model.to(device)


#express_ind = [ i for i in range(50)]

x_data = val_X[express_ind]
y_data = val_Y[express_ind,0]

x_data_tor = Variable(torch.tensor(x_data).float().to(device) , requires_grad=True)
y_data_tor = torch.tensor(y_data).to(device)

new_x_data = []
for img in x_data:
    tmp = []
    for d in img[0].reshape(-1):
        for i in range(3):
            tmp.append(d)
    tmp = np.array(tmp).reshape(48,48,3)
    new_x_data.append(tmp)
x_data = np.array(new_x_data)

print('x data shape {}'.format(x_data.shape))
print('y data shape {}'.format(y_data.shape))
print(y_data)

def predict(data):
    data = torch.from_numpy(data).float().to(device)
    data = Variable(data, requires_grad=True)
    data = data[:,:,:,0].reshape(-1,1,48,48)
    pred = model(data).detach().cpu().numpy()
    return pred

def segmentation(data):
    # Input: image numpy array
    # Returns a segmentation function which returns the segmentation labels array ((48,48) numpy array)
    return slic(data)
    # TODO:
    # return ?
explainer = lime_image.LimeImageExplainer()

for idx in range(7):
    # Initiate explainer instance
    # Get the explaination of an image
    y_pred = torch.max(model(x_data_tor[idx].view(-1,1,48,48)),1)[1]
    
    y_pred = y_pred.detach().cpu().numpy()[0]
    #print(type(y_pred))
    #print(y_pred.shape)

    explaination = explainer.explain_instance(image=x_data[idx], 
                                classifier_fn=predict,
                                segmentation_fn=segmentation
                            )
    # Get processed image
    #if not os.path.exists('lime_images/'+str(y_data[idx])):
            #os.makedirs('lime_images/'+str(y_data[idx]))

    #plt.imsave('lime_images/'+str(y_data[idx]) +'/'+str(idx)+'_ori.png',x_data[idx])
    pred_image, mask = explaination.get_image_and_mask(label=y_pred,positive_only=False,hide_rest=False,num_features=5,min_weight=0.0)
    #plt.imsave('lime_images/'+str(y_data[idx]) +'/'+str(idx)+'_pred'+str(y_pred)+'.png', pred_image)
    try:
        image, mask = explaination.get_image_and_mask(label=y_data[idx],positive_only=False,hide_rest=False,num_features=5,min_weight=0.0)
        plt.imsave(args.path + 'fig3_'+str(idx) +'.jpg', image)
    except:
        print('img_'+str(idx)+' is not predicted right')
    # save the image
    

    
    