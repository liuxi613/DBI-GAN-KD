import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from logi import LogisticRegression as student_net
import numpy as np
from getministac import Net 
# Training settings
batch_size = 64
#img_transform = transforms.Compose([
#        transforms.ToTensor(), 
#        transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))
#    ])

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

if __name__ =="__main__":
    model=student_net(784,10)
    model1=student_net(784,10)
    teacher = Net()
    model1.load_state_dict(torch.load("./class_logi_add.pth"))
    teacher.load_state_dict(torch.load("./class.pth"))
    model.load_state_dict(torch.load("./class_logi.pth"))
    correct = 0
    correct1=0
    correctt=0
    zongshu = 0
    for data, target in test_loader:
        da=[]
        tar=[]
        #data, target = Variable(data, volatile=True), Variable(target)
        data=data.numpy().tolist()
        target=target.numpy().tolist()

        for i in range(len(target)):
            if(target[i]==1 or target[i]==2):
                da.append(data[i])
                tar.append(target[i])
        zongshu= zongshu +len(tar)
        da = torch.Tensor(np.array(da))
        tar =  torch.Tensor(np.array(tar))
        da1 = Variable(da, volatile=True)
        da, tar = Variable(da.view(-1, 28 * 28), volatile=True), Variable(tar)
        tea_out = teacher(da1)
        t_pred = tea_out.data.max(1, keepdim=True)[1]
        output = model(da)
        pred = output.data.max(1, keepdim=True)[1]
        output1 = model1(da)
        pred1 = output1.data.max(1, keepdim=True)[1]
        correct1 += pred1.eq(tar.long().data.view_as(pred1)).cpu().sum()
        correct +=pred.eq(tar.long().data.view_as(pred)).cpu().sum()
        correctt += t_pred.eq(tar.long().data.view_as(t_pred)).cpu().sum()
    print('Accuracy:({:.0f}%)\n'.format(100. * correct/zongshu ))
    print('Accuracy:({:.0f}%)\n'.format(100. * correct1/zongshu ))
    print('Accuracy:({:.0f}%)\n'.format(100. * correctt/zongshu ))

