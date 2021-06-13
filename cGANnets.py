import torch
import  torch.nn as nn
import torch.functional as F


#变成CGAN 在fc层嵌入 one-ho编码

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,5),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2,stride = 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5,padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64*7*7+10,1024),
            nn.LeakyReLU(0.2,True),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )

    def forward(self, x,labels):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = torch.cat((x,labels),1)
        
        x = self.fc(x)
        return x

class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size+10, num_feature)  # batch, 3136=1x56x56
        self.br = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.downsample1 = nn.Sequential(
            nn.Conv2d(3,50,3,stride=1,padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )

        self.downsample2 =  nn.Sequential(
            nn.Conv2d(50,25,3,stride=1,padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )

        self.downsample3 = nn.Sequential(
            nn.Conv2d(25,3,2,stride = 2),
            nn.Tanh()
        )

    def forward(self,z,labels):
        '''

        :param x: (batchsize,100）的随机噪声
        :param label:  (batchsize,10) 的one-hot 标签编码
        :return:
        '''
        x = torch.cat((z,labels),1) #沿1维拼接
        x = self.fc(x)

        x = x.view(x.size(0),3,64,64)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)

        return x
