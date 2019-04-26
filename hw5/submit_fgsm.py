import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50,resnet101, densenet121, densenet169 
from scipy.misc import imsave
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model", default='resnet50',type=str)
parser.add_argument("-epsilon", default=10/255,type=float)
parser.add_argument("-in_path",type=str)
parser.add_argument("-out_path",type=str)
args = parser.parse_args()

print(args)

device = torch.device('cuda')
# using pretrain proxy model, ex. VGG16, VGG19...
if args.model=='resnet50':
    model = resnet50(pretrained=True)
elif args.model=='vgg16':
    model = vgg16(pretrained=True)
elif args.model=='vgg19':
    model = vgg19(pretrained=True)
elif args.model=='resnet101':
    model = resnet101(pretrained=True)
elif args.model=='densenet121':
    model = densenet121(pretrained=True)
elif args.model=='densenet169':
    model = densenet169(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.eval()



# loss criterion
loss_fn = nn.CrossEntropyLoss()

file = pd.read_csv('labels.csv')

imgs = file['ImgId'].values.astype(np.float)
lbls = file['TrueLabel'].values

print(imgs)
print(lbls)

epsilon = args.epsilon

attack_lbls = []
model_lbls = []

for i in range(imgs.shape[0]):
    if imgs[i] < 10:
        imgname = args.in_path+'/00'+ str(int(imgs[i])) + '.png'
    elif imgs[i] < 100:
        imgname = args.in_path+'/0'+ str(int(imgs[i])) + '.png'
    else:
        imgname = args.in_path+'/'+ str(int(imgs[i])) + '.png'
    print(imgname)

    img = Image.open(imgname)
    # you can do some transform to the image, ex. ToTensor()
    trans = transform.Compose([transform.ToTensor()])
    
    img = trans(img)
    img = img.unsqueeze(0)
    #img = Variable(img,device=device,requires_grad=True)
    img.requires_grad = True
    # set gradients to zero
    
    zero_gradients(img)
    pred = model(img)

    target = torch.Tensor([lbls[i]]).long()

    myloss = (-1)*loss_fn(pred, target)
    myloss.backward() 

    label = torch.max(pred,1)[1]
    model_lbls.append(label[0])
    
    # add epsilon to image
    img = img - epsilon * img.grad.sign_()

    pred = model(img)
    label = torch.max(pred,1)[1]
    attack_lbls.append(label[0])
    # do inverse_transform if you did some transformation
    output_file = args.out_path+'/'+ imgname[-7:]

    img = img.squeeze().detach().numpy()

    img = np.rollaxis(img, 2) 
    img = np.rollaxis(img, 2) 
    #image = inverse_trasform(image) 
    imsave(output_file, img)


model_lbls = np.array(model_lbls)
print(lbls)
print(model_lbls)
acc = np.mean((lbls == model_lbls))
print('model acc {}'.format(acc))

attack_lbls = np.array(attack_lbls)
print(lbls)
print(attack_lbls)
acc = np.mean((lbls == attack_lbls))
print('attack acc {}'.format(acc))

acc = np.mean((model_lbls == attack_lbls))
print('attack model acc {}'.format(acc))
