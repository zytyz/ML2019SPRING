# standard library
import argparse
import csv
import time
import sys
import os
import glob
# other library
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# PyTorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data 
from multiprocessing import Pool
from torch.utils.data import DataLoader

class Dataset(data.Dataset):
    def __init__(self, image_dir):

        files = [f for f in glob.glob(image_dir + '*.jpg', recursive=True)]
        files.sort()
        print('total image number: {}'.format(len(files)))  
        p = Pool()

        total_img = p.map(self.getFileArray,files)
        total_img= np.array(total_img, dtype=float)
        # since at pytorch conv layer, input=(N, C, H, W)
        self.total_img = np.transpose(total_img, (0, 3, 1, 2))
        # normalize
        self.total_img = (self.total_img ) / 255.0
        print("=== total image shape:",  self.total_img.shape)
        # shape = (40000, 3, 32, 32)

    def getFileArray(self,filename):
        return np.array(Image.open(filename))

    def __len__(self):
        return len(self.total_img)

    def __getitem__(self, index):
        return(self.total_img[index])

class Net(nn.Module):
    def __init__(self, image_shape, latent_dim):
        super(Net, self).__init__()
        self.shape = image_shape
        self.latent_dim = latent_dim
        self.modelnum = args.modelnum

        if args.modelnum in [0,1,2,3,4,5,6,7,8]:
            convfilters = [16,32,32]
        elif args.modelnum in [9]:
            convfilters = [32,64,128]
        elif args.modelnum in [10]:
            convfilters = [64,128,256]

        self.encoder = nn.Sequential(
            nn.Conv2d(3, convfilters[0], 3, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(convfilters[0], convfilters[1], 3, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(convfilters[1], convfilters[2], 3, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.MaxPool2d(2, 2)
        )
        self.encode_shape = np.array([convfilters[2],4,4]).astype(np.float)
        N = int(np.prod(self.encode_shape))
        print('encode shape {}'.format(self.encode_shape))
        print('N: {}'.format(N))
            # assume output shape is (Batch, 16, 4, 4)
            

        if args.modelnum in [0,1,2,3,4,5,6,7]:
            assert self.latent_dim <= N
            self.fc1 = nn.Linear(N, self.latent_dim)
            self.fc2 = nn.Linear(self.latent_dim, N)
        elif args.modelnum in [8,9]:
            pass

        self.decoder = nn.Sequential(
           # TODO: define yout own structure
            nn.Conv2d(convfilters[2], convfilters[1], 3, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(convfilters[1], convfilters[0], 3, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(convfilters[0], 3, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        x = self.encoder(x)
        #print('after encoder {}'.format(x.shape))
        # flatten
        x = x.view(len(x), -1)
        if self.modelnum in [0,1,2,3,4,5,6,7]:
            encoded = self.fc1(x)
        elif self.modelnum in [8,9,10]:
            encoded = x
        #print('encoded --> Linear --> latent_dim {}'.format(encoded.shape))
        if self.modelnum in [0,1,2,3,4,5,6,7]:
            x = F.relu(self.fc2(encoded))
        elif self.modelnum in [8,9,10]:
            x = encoded
        #print('latent_dim --> Linear --> decoded {}'.format(x.shape))

        decode_shape = np.ndarray.tolist(self.encode_shape.astype(np.int))
        x = x.view(-1, decode_shape[0], decode_shape[1], decode_shape[2])
        
        #print('before decoder {}'.format(x.shape))
        x = self.decoder(x)
        return encoded, x

def training(train, val, model, device, n_epoch, batch_size, save_name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    global time
    print('=== start training, parameter total:%d, trainable:%d' % (total, trainable))
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_loss = 100

    for epoch in range(n_epoch):
        total_loss = 0

    # training set
        model.train()
        for batchidx in range(int(len(train)/batch_size)+1):
            #image is a batch
            image = train[batchidx*batch_size:(batchidx+1)*batch_size]
            image = image.to(device, dtype=torch.float)
            _, reconstruct = model(image)
            loss = criterion(reconstruct, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            print('[Epoch %d | %d/%d] loss: %.10f' %
                 ((epoch+1), batchidx*batch_size, len(train), loss.item()), end='\r')
        total_loss /= len(train)
        print("\n  Training  | Loss: {}".format(total_loss))

    # validation set
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batchidx in range(int(len(val)/batch_size)+1):
                image = val[batchidx*batch_size:(batchidx+1)*batch_size]
                image = image.to(device, dtype=torch.float)
                _, reconstruct = model(image)

                loss = criterion(reconstruct, image)
                total_loss += loss.item() 
            total_loss /= len(val)
            print(" Validation | Loss: {} ".format(total_loss))
        # save model
        if total_loss < best_loss:
            print('val loss improved from {} to {}'.format(best_loss,total_loss))
            best_loss = total_loss
            print("saving model with loss %.20f...\n" % total_loss)
            torch.save(model,save_name)
            time = 0
        else:
            time+=1
            print('val loss did not improve')
            if time >=5:
                print('early stopping')
                break

def mydatasplit(args,totaldata):
    try:
        with open('val_ind/ind_list'+str(args.modelnum)+'.txt') as f:
            ind_list = f.read().split(' ')
            ind_list = [int(x) for x in ind_list]
    except:
        ind_list = [ i for i in range(totaldata.shape[0])]
        import random as rd
        rd.shuffle(ind_list)
    print('ind list {}'.format(ind_list[:10]))
    val_len = int( 1/20 * totaldata.shape[0])

    path = 'val_ind/ind_list'+str(args.modelnum)+'.txt'
    with open(path,'w') as f:
        print('writing ind list in '+path)
        f.write(' '.join([str(x) for x in ind_list]))
    #print('total data length {}'.format(totaldata.shape[0]))
    val_data = totaldata[ind_list[:val_len]]
    train_data = totaldata[ind_list[val_len:]]
    return train_data, val_data

def clustering(model, device, loader, n_iter, reduced_dim):
    model.eval()
    #latent_vec = np.array([])
    for idx, image in enumerate(loader):
        print("predict %d / %d" % (idx, len(loader)) , end='\r')
        image = image.to(device, dtype=torch.float)
        latent, r = model(image)
        try:
            latent_vec = np.concatenate((latent_vec, latent.cpu().detach().numpy()),axis=0)
        except:
            latent_vec = latent.cpu().detach().numpy()
    #latent_vec = latent_vec.cpu().detach().numpy()
    print('\n')
    print(latent_vec.shape)
    # shape = (40000, latent_dim)
    if reduced_dim!=-1:
        pca = PCA(n_components=reduced_dim, copy=False, whiten=True, svd_solver='full')
        latent_vec = pca.fit_transform(latent_vec)
        print(latent_vec.shape)

    kmeans = KMeans(n_clusters=2, random_state=0, max_iter=n_iter).fit(latent_vec)
    return kmeans.labels_

def read_test_case(path):
    dm = pd.read_csv(path)
    img1 = dm['image1_name']
    img2 = dm['image2_name']
    test_case = np.transpose(np.array([img1, img2]))
    return test_case

def prediction(label, test_case, output):
    result = []
    for i in range(len(test_case)):
        index1, index2 = int(test_case[i][0])-1, int(test_case[i][1])-1
        if label[index1] != label[index2]:
            result.append(0)
        else:
            result.append(1)
    
    result = np.array(result)
    print('saving ans to {}'.format(output))
    with open(output, 'w') as f:
        f.write("id,label\n")
        for i in range(len(test_case)):
            f.write("%d,%d\n" % (i, result[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num','--modelnum',default=0,type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--image_dir', default='/home/mlgogo/Desktop/hw7/data/images/', type=str)
    parser.add_argument('--test_case', default='data/test_case.csv', type=str)
    parser.add_argument('--output_name', default='output_default.csv', type=str)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--reload',default=-1, type=int)
    parser.add_argument('--train', default=False, type=bool)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda')
    dataset = Dataset(image_dir=args.image_dir)


    if args.modelnum==0:
        reduced_dim, latent_dim = 16,128
    elif args.modelnum==1:
        reduced_dim, latent_dim = 32,128
    elif args.modelnum==2:
        reduced_dim, latent_dim=-1,128
    elif args.modelnum==3:
        reduced_dim, latent_dim=64,128
    elif args.modelnum==4:
        reduced_dim, latent_dim=128,128
    elif args.modelnum==5:
        reduced_dim, latent_dim=200,256
    elif args.modelnum==6:
        reduced_dim, latent_dim=128,256
    elif args.modelnum==7:
        reduced_dim, latent_dim=200,256
    elif args.modelnum==8:
        reduced_dim, latent_dim = 512,None
    elif args.modelnum==9:
        reduced_dim, latent_dim = 512,None
    elif args.modelnum==10:
        reduced_dim, latent_dim = 1024,None
    elif args.modelnum==11:
        reduced_dim, latent_dim = -1,None


    if args.train==True:
        model = Net(image_shape=(3,32,32),latent_dim=latent_dim)
        model.to(device)

        train_data, val_data = mydatasplit(args,dataset.total_img)
        train_data = torch.from_numpy(train_data)
        val_data = torch.from_numpy(val_data)

        print('train data shape {}'.format(train_data.shape))
        print('val data shape {}'.format(val_data.shape))

        time = 0
        training(train_data,val_data,model=model,device=device,n_epoch=100,batch_size=128,save_name='torchckpt/ckpt'+str(args.modelnum)+'.pkl')

        del model

    if args.reload==-1:
        args.reload=args.modelnum
    model = torch.load('torchckpt/ckpt'+str(args.reload)+'.pkl')
    model.to(device)

    loader = DataLoader(dataset=dataset, batch_size=128, shuffle=False)
    print('clustering...')

    labels = clustering(model, device, loader, n_iter=200, reduced_dim=reduced_dim)
    print(labels.shape)
    print(labels)

    test_case = read_test_case(args.test_case)
    print(test_case.shape)

    prediction(label=labels, test_case=test_case, output=args.output_name)















