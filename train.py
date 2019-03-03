import sys
import os
import csv
from optparse import OptionParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms

from dataloader import DataLoader
from common import show_pred_image, save_pred_image
from model import ResNet

def train_net(net, epochs=5, lr=0.001, data_root='data/', save_cp=True, gpu=True):

    if gpu:
        num_workers = 0
    else:
        num_workers = 0

    train_loader = DataLoader(data_root, 'train')
    train_loader.setMode('train')
    train_data_loader = torch.utils.data.DataLoader(train_loader,
                                                    batch_size=8,
                                                    shuffle=True,
                                                    num_workers=num_workers)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()

        epoch_loss = 0

        for i, (data_input, label, img_name) in enumerate(train_data_loader):
            # shape = data_input.shape
            # print("in training -  itr", itr, "i", i, "shape", shape)
            
            # torch to float
            input_torch = data_input.float()
            label_torch = label.float()

            # load image tensor to gpu
            if gpu:
                input_torch = Variable(input_torch.cuda())
                label_torch = Variable(label_torch.cuda())
            else:
                input_torch = Variable(input_torch)
                label_torch = Variable(label_torch)

            # get predictions 
            optimizer.zero_grad()
            pred_torch = net(input_torch)
            
            # todo: get prediction and getLoss()
            loss = criterion(pred_torch, label_torch)

            # optimize weights
            loss.backward()
            optimizer.step()

            # record loss
            epoch_loss += loss.item()
            print('Epoch %d | Iteration %d - Loss: %.6f' % (epoch+1, i+1, loss.item()))

            # save trained image at end of epoch
            # for i in range(8):
            print(img_name[0], pred_torch[0].detach().cpu(), label[0])

        
        # save model when necessary
        if (epoch+1)%10==0:
            torch.save(net.state_dict(), 'output/epoch%d.pth' % (epoch + 1))
            print('Checkpoint %d saved !' % (epoch + 1))
        
        # show loss per epoch
        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / (i + 1)))

    
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int', help='number of epochs')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')

    (options, _) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = ResNet()

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net, epochs=args.epochs, gpu=args.gpu)
