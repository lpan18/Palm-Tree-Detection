import sys
import os
from optparse import OptionParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import csv

from model import ResNet
from dataloader import DataLoader
from common import show_pred_image, save_pred_image

def test_net(net, model='CP20.pth', data_root='data/', gpu=True):

    if gpu:
        num_workers = 0
    else:
        num_workers = 0

    test_loader = DataLoader(data_root, 'test')
    test_data_loader = torch.utils.data.DataLoader(test_loader,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=num_workers)
    
    # displays test images with original and predicted masks after training
    test_loader.setMode('test')
    net_state = torch.load('output/' + model)
    net.load_state_dict(net_state)
    net.eval()

    results = []

    with torch.no_grad():
        for i, (data_input, _, img_name) in enumerate(test_data_loader):

            # torch to float
            input_torch = data_input.float()

            # load image tensor to gpu
            if gpu:
                input_torch = Variable(input_torch.cuda())
            else:
                input_torch = Variable(input_torch)

            # run net
            pred_torch = net(input_torch)

            # save result
            results.append([img_name[0], pred_torch[0].detach().cpu().numpy()[0]])

    with open('result.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'has_oilpalm'])
        writer.writerows(results)
    
def get_args():
    parser = OptionParser()
    parser.add_option('-m', '--model', dest='model', default='epoch10.pth', help='output directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')

    (options, _) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = ResNet()

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    test_net(net=net, model=args.model, gpu=args.gpu)
