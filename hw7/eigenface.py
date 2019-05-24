import os
import sys
import numpy as np 
from skimage.io import imread, imsave
import glob
from multiprocessing import Pool
import sys

IMAGE_PATH = sys.argv[2]

# Images for compression & reconstruction
test_image = []
test_image.append(sys.argv[3]) 

# Number of principal components used

def process(M):
    M = np.copy(M) 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

def readimage(fname):
	return imread(os.path.join(IMAGE_PATH,fname)).flatten()

filelist = [os.path.split(f)[1] for f in glob.glob(IMAGE_PATH + '*.jpg', recursive=True)]
filelist.sort()
print(filelist[:10])
print('file len :',len(filelist))

# Record the shape of images
img_shape = imread(os.path.join(IMAGE_PATH,filelist[0])).shape 

p = Pool()
img_data = p.map(readimage,filelist)

training_data = np.array(img_data).astype('float32')

# Calculate mean & Normalize
mean = np.mean(training_data, axis = 0)  
training_data -= mean 

print(training_data.shape)

average = process(np.copy(mean))
imsave('average.jpg', average.reshape(img_shape)) 
print('average: ',average)

# Use SVD to find the eigenvectors 
if sys.argv[1]=='svd':
	print('doing svd...')
	u, s, v = np.linalg.svd(np.transpose(np.copy(training_data)), full_matrices = False) 
	np.save('u.npy',u)
	np.save('s.npy',s)
	np.save('v.npy',v) 
elif sys.argv[1]=='load':
	u = np.load('u.npy')
	s = np.load('s.npy')
	v = np.load('v.npy')
print('u shape {}'.format(u.shape))
print('s shape {}'.format(s.shape))
print('v shape {}'.format(v.shape))

print(np.dot(np.transpose(u)[0],np.transpose(u)[1]))
print(np.dot(np.transpose(u)[0],np.transpose(u)[0]))


for x in range(10):
    eigenface = process(np.transpose(u)[x])
    imsave('eigenfaces/'+str(x) + '_eigenface.jpg', eigenface.reshape(img_shape))  

k = 5

print(np.dot(np.transpose(u)[0],np.transpose(u)[1]))
print(np.dot(np.transpose(u)[0],np.transpose(u)[0]))
#print(np.allclose(training_data+mean,training_data_tmp))
#imsave('shit_0_2.jpg', (training_data[0]+mean).reshape(img_shape).astype(np.uint8)) 

for x in test_image: 
    # Load image & Normalize
    picked_img = imread(IMAGE_PATH + x)  
    X = picked_img.flatten().astype('float32') 
    X -= mean
    # Compression
    weight = np.array([np.dot(X,np.transpose(u)[i]) for i in range(k)]) 
    print(weight)
    print('weight shape',weight.reshape(1,-1).shape)
    print('u',u[:,:5].shape)
    # Reconstruction
    reconstruct = np.matmul(weight.reshape(1,-1),np.transpose(u)[:k]).reshape(-1)
    reconstruct = reconstruct + mean
    reconstruct = np.clip(reconstruct,0.0,255.0).astype(np.uint8)
    print(reconstruct)
    imsave(sys.argv[4], reconstruct.reshape(img_shape)) 

lamda = np.square(np.copy(s))
print('lamda')
for i in range(k):
    ratio = lamda[i]/lamda.sum()
    print('k {}, ratio {}'.format(i,ratio))

print('s')
for i in range(k):
    ratio = s[i]/s.sum()
    print('k {}, ratio {}'.format(i,ratio))
