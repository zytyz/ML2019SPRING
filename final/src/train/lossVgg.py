import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms



class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()

        self.mse = nn.MSELoss(size_average=True).cuda()
        vgg16 = models.vgg16(pretrained=True)
        vgg16 = vgg16.cuda()
        for param in vgg16.parameters():
            param.requires_grad = False
            
        self.feature_ext = vgg16.features[:12]
        for param in self.feature_ext.parameters():
            param.requires_grad = False

        del vgg16

        '''self.transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])'''
    def trans(self,img):
        img2 = img[:,:,:224,:224]
        img3 = ( img2 - 0.449 )/0.226
        return img3

    
    def forward(self, hazy_img, clear_img):
        hazy_img = self.trans(hazy_img)
        clear_img = self.trans(clear_img)

        hazyvgg = self.feature_ext(hazy_img)
        clearvgg = self.feature_ext(clear_img)
        loss = self.mse(hazyvgg, clearvgg)
        return loss
