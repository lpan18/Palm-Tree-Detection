import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import DataLoader
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        ResNet50 = models.resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        
        self.backbone = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, 32)
        # self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        out = self.backbone(x)
        # print('backbone', out.shape)
        out = out.view(out.size(0), -1)
        # print('view', out.shape)
        out = self.fc1(out)
        # print('fc1',out.shape)
        # out = self.dropout(out)
        out = self.fc2(out)
        # print('fc2',out.shape)
        out = self.sigmoid(out)

        return out

if __name__ == '__main__':
    loader = DataLoader('data/', 'train')
    train_data_loader = torch.utils.data.DataLoader(loader,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0)
    idx, (image, label) = next(enumerate(train_data_loader))
    net = ResNet()
    net.forward(image)