import math
from random import random

import numpy as np
import torch
from PIL import Image
from torch.utils import data
import  os
from torch.utils import data
from torchvision import transforms as T

listpy="/home/smhe/megaage/megaage"
class MegaageDataset(data.Dataset):
    def __init__(self, transform,mode):


        self.transform = transform
        self.mode=mode
        self.L=3
        self.train_dataset = []
        self.test_dataset = []
        self.alpha=0
        self.crop_size=[71,-51]
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        self.finx=["_age.txt","_name.txt","_dis.txt"]

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)
    def preprocess(self):

        dataset_dict= {
            'train':self.train_dataset,
            'test':self.test_dataset
        }
        for mode in ['train','test']:
            fame_list= [line.rstrip() for line in open(os.path.join(listpy,'list',mode+self.finx[0]), 'r')]
            label_list = [line.rstrip() for line in open(os.path.join(listpy,'list',mode+self.finx[1]), 'r')]
            aplha_list = [line.rstrip() for line in open(os.path.join(listpy,'list',mode+self.finx[2]), 'r')]

            for i, line in enumerate(fame_list):
                filename = line.split()
                label=int(label_list[i].split())
                aplha=aplha_list[i].split()
                aplha=[np.float64(num) for num in aplha]


                dataset_dict[mode].append([filename,label, aplha])

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label, aplha = dataset[index]
        image = Image.open(os.path.join(listpy,self.mode, filename))
        image = np.array(image)
        image = image[self.crop_size[0]:self.crop_size[1], self.crop_size[2]:self.crop_size[3]]
        image = Image.fromarray(image.astype('uint8')).convert('RGB')
        label=label-1
        one=torch.IntTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        one[label:]=1
        cost_one=torch.IntTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        cost_one[label-self.L:label+self.L]=0
        if self.alpha==0:
            x_01 = np.linspace(1,70, 70)
            y_sig01 = np.exp(-(x_01 - label) ** 2 / (2 * self.alpha ** 2)) / (math.sqrt(2 * math.pi) * self.alpha)
            y_sig01=torch.tensor(y_sig01)
        else:
            y_sig01=aplha
        return self.transform(image), torch.FloatTensor(one),torch.FloatTensor(cost_one),torch.FloatTensor(y_sig01)
def get_loader(image_dir, attr_path, selected_attrs, crop_size=(218,178), image_size=128,
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())

    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)


    dataset = MegaageDataset( transform, 'train')
    datatest = MegaageDataset(transform, 'test')

    test_loader = data.DataLoader(dataset=datatest,
                                  batch_size=1,
                                  shuffle=(mode=='test'),
                                  num_workers=1)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader,test_loader
