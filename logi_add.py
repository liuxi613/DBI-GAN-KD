import torch
import torch.nn as nn
import torch.nn.functional as F
#from getministac import Net
from torchvision import datasets
#from skimage import io,transform
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
from PIL import Image
from torchvision import transforms, utils
import torchvision
import numpy as np
import torch.utils.data as Data
import cGANnets as GANnet
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#import torchvision.datasets as dsets
BATCH_SIZE = 128
batch_size = 128
n_iters = 500
temperature = 3

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

#test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_test)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE, shuffle=False)

minist = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    #print(minist)
    #data=minist['data']
    #labels=train_data['target'].reshape(-1,1)
    #print(data,labels)

train_loader = torch.utils.data.DataLoader(
        dataset = minist,batch_size=BATCH_SIZE,shuffle =True,
        drop_last = True
    )

class seaNet(torch.nn.Module):
    def __init__(self):
        super(seaNet, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3)
        self.mp = torch.nn.MaxPool2d(2)
        # fully connect
        self.fc = torch.nn.Linear(10*15*15, 10)

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 128*10*15*15
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*20*7*7
        #x = F.relu(self.mp(self.conv2(x)))
        # x: 64*320
        x = x.view(in_size, -1) # flatten the tensor
        # x: 64*10
        x = self.fc(x)
        #return F.softmax(x)
        return x
'''
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(720, 2)

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 128*10*15*15
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*20*7*7
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*320
        x = x.view(in_size, -1) # flatten the tensor
        # x: 64*10
        x = self.fc(x)
        #return F.softmax(x)
        return x
'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


epochs = n_iters / (len(train_loader) / 256)
input_dim = 784
output_dim = 10
lr_rate = 0.0001

tmodel = ResNet18()
smodel= seaNet()
tmodel.cuda()
smodel.cuda()
tmodel.load_state_dict(torch.load("./class.pth"))
smodel.load_state_dict(torch.load("./class_logi_add.pth"))

criterion_CE = nn.CrossEntropyLoss()
z_dimension = 100 



def generatelabels(batchsize,real_labels =None):
    x = torch.Tensor(torch.zeros(batchsize,10)).to(device)
    if real_labels is None: #生成随机标签
        y = [np.random.randint(0, 9) for i in range(batchsize)]
        '''
        y=[]
        for i in range(batchsize):
            tmp=np.random.randint(0,9)
            if(tmp==2):
                tmp=1
            y.append(tmp)
        '''
        x[np.arange(batchsize), y] = 1
    else:
        x[np.arange(batchsize),real_labels] = 1

    return x

#criterion = torch.nn.MSELoss()
#aoptimizer = torch.optim.Adam(smodel.parameters(), lr=lr_rate)
iter = 0
if __name__ =="__main__":
    G = GANnet.generator(z_dimension,12288)
    G.cuda()
    G.load_state_dict(torch.load("./yuanshi_generator.pth"))

    #model.load_state_dict(torch.load("./class_logi.pth"))
    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(smodel.parameters(), lr=lr_rate)
       
    #ateacher=Net()
    #teacher.load_state_dict(torch.load("./class.pth"))
    for epoch in range(int(epochs)):
        if epoch < 1/4*epochs:
            #optimizer = optim.SGD(smodel.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            optimizer = torch.optim.Adam(smodel.parameters(), lr=0.001)
        elif epoch < 1/2*epochs:
            #optimizer = optim.SGD(smodel.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
            optimizer = torch.optim.Adam(smodel.parameters(), lr=0.0001)
        elif epoch < 4/5*epochs:
            #optimizer = optim.SGD(smodel.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
            optimizer = torch.optim.Adam(smodel.parameters(), lr=0.00001)


        for i, (imgs, real_labels) in enumerate(train_loader):
            imgs = Variable(imgs).cuda()
            real_labels = Variable(real_labels.long()).view(1,-1)[0]
            real_labels =real_labels.cuda()
            out_t = tmodel(imgs)
            out_s = smodel(imgs)
            ratio = max(3 * (1 - epoch / epochs), 0) + 1

            loss = criterion_CE(out_s, real_labels)
            loss += - ratio * (F.softmax(out_t, 1).detach() * F.log_softmax(out_s, 1)).sum() / imgs.size(0)
            
            z = torch.Tensor(torch.randn(400, z_dimension)).to(device)
            fake_labels = generatelabels(400,0) # 生成编码标签
            fake_labels.requires_grad = True
            ratio_attack = max(2 * (1 - 4 / 3 * epoch / epochs), 0) + 0
            fake_img = G(z, fake_labels)
            attack_out_t = tmodel(fake_img)
            sss=F.softmax(attack_out_t,1)
            re,re1=sss.sort(1,descending=True)

            attack_out_s = smodel(fake_img)
            k=1
            loss1=0
            for i in range(len(attack_out_t)):
                 if (abs(sss[i][re1[i][0]]-sss[i][re1[i][1]])<0.2 and sss[i][re1[i][0]]>=0.4):

                      k=k+1
                      loss1 += - ratio_attack * (F.softmax(attack_out_t[i] / temperature).detach() * F.log_softmax(attack_out_s[i] / temperature)).sum() #/ 64
                      #print(F.softmax(attack_out_t[i]))
            loss=loss+loss1/k
            
            #print(k)
            optimizer.zero_grad()
            #print(loss)
            loss.backward()
            optimizer.step()
        total=0
        correct = 0
             
        for images, labels in test_loader:
            images = Variable(images).cuda()#.view(-1, 3*32*32))
            outputs = smodel(images)
            #print(labels)
            labels = Variable(labels).view(1,-1)[0].long().cuda()
            _, predicted = torch.max(outputs.data, 1)
                #print(predicted,labels)
            total+= labels.size(0)
                # 如果用的是 GPU，则要把预测值和标签都取回 CPU，才能用 Python 来计算
            correct+= (predicted == labels).sum()
        accuracy = 100. * float(correct)/total
        print("Iteration: {}. Loss: {}. Accuracy: {}.".format(i, loss.item(), accuracy))
        torch.cuda.empty_cache()
    torch.save(smodel.state_dict(), './class_logi_add.pth')
