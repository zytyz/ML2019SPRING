from os import listdir
from os.path import join
from PIL import Image
from os.path import basename
import torch.utils.data as data
import random
import numpy as np
from scipy.ndimage import rotate

def is_image_file(filename):
  filename_lower = filename.lower()
  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=None, phase='train'):
        super(DatasetFromFolder, self).__init__()

        data_dir = "%s/data/" % image_dir
        label_dir = "%s/label/" % image_dir
        self.data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        self.label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]

        self.transform = transform
        self.phase = phase

    def __getitem__(self, index):
        data = np.asarray(Image.open(self.data_filenames[index]))
        label = np.asarray(Image.open(self.label_filenames[index]))

        if self.phase=='train':
            num = random.randint(0,11)
            flip = num//4
            degree = num%4

            if flip == 1:
                data = np.flip(data, 0)
                label = np.flip(label, 0)
            if flip == 2:
                data = np.flip(data, 1)
                label = np.flip(label, 1)

            if degree != 0:
                data = rotate(data, 90 * degree)
                label = rotate(label, 90 * degree)
        elif self.phase == 'test':
            pass

        if self.transform:
            data = self.transform(data.copy())
            label = self.transform(label.copy())

        #return data.half(), label.half()
        return data, label

    def __len__(self):
        return len(self.data_filenames)
