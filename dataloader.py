import os
import random
import numpy as np
from common import show_image
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class DataLoader():
    def __init__(self, root_dir='data/', mode='train'):
        '''
            Function to load the dataset list
            :param root_dir [str] - parent folder
        '''
        self.root_dir = os.path.abspath(root_dir)
        self.mode = mode
        if self.mode == 'test':
            self.data_dir = os.path.join(self.root_dir, 'test')
            self.files = os.listdir(self.data_dir)

        else:
            self.data_dir = os.path.join(self.root_dir, 'train')

            df = pd.read_csv('./data/traininglabels.csv')
            self.has_palm_labels = df[df['has_oilpalm']==1]
            temp = df[df['has_oilpalm']==0]
            self.no_palm_labels = temp[temp['score']>0.75]

            self.get_random_no_palm()

    def __len__(self):
        '''
            Function to get length of the dataset list
        '''
        if self.mode == 'train':
            return len(self.has_palm_labels)*2
        return len(self.files)
    

    def setMode(self, mode):
        '''
            Function to set mode of the dataset list
        '''
        self.mode = mode


    def augment_data(self, img_original):
        img_w, img_h = img_original.size
        img_l = min(img_w, img_h)
        img_h = max(img_w, img_h)
        how_large = random.randint(np.floor(img_l / 4), img_h)
        methods = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.5)
        ])
        return methods(img_original)


    def get_random_no_palm(self):
        length = len(self.has_palm_labels)
        idx = random.sample(list(range(length)), length)
        no_palm = []
        dict = self.no_palm_labels.to_dict('records')
        for i in idx:
            no_palm.append(dict[i])
        # print(no_palm)
        self.label = self.has_palm_labels.to_dict('records')+no_palm

    def __getitem__(self, idx):
        '''
            Function to get one item from the dataset list
            :param idx [int] - index of item to get
        '''
        if self.mode == 'train':  
            if self.label[idx]['has_oilpalm'] == 0:
                label = 0.0
            else:
                label = self.label[idx]['score']

            img_name = self.label[idx]['image_id']
            
            item_path = os.path.join(self.data_dir, img_name)

        else:
            item_path = os.path.join(self.data_dir, self.files[idx])
            label = 0.0
            img_name = self.files[idx]

        img_original = Image.open(item_path).convert('RGB')

        # data augmentation
        if self.mode == 'train':
            img_original = self.augment_data(img_original)
            r = random.randint(-2, 2)
            img_original = transforms.functional.rotate(img_original, 90*r)

        # resize img to 388 and add padding; label resize to 128
        resized_size = 224
        img_original = transforms.functional.resize(img_original, [resized_size, resized_size])

        # PIL to torch, normalize the image
        original_torch = transforms.functional.to_tensor(img_original)

        # show_image(original_torch, is_tensor=True)
        
        return (original_torch, label, img_name)


if __name__ == '__main__':
    loader = DataLoader('data/', 'train')
    train_data_loader = torch.utils.data.DataLoader(loader,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0)
    idx, (image, label) = next(enumerate(train_data_loader))


