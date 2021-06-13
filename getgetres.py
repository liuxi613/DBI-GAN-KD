import torch
import  torch.nn as nn
import torch.functional as F
import os
from tensorboardX import SummaryWriter
import torchvision
from torchvision import datasets
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image,make_grid
import cGANnets as GANnet
from train import train_1,train_2
from getministac import Net
from logi import LogisticRegression
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('agg')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import argparse
import scipy.io as io

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchsize = 128


z_dimension = 100
num = 25
epoch_num  = 1000
scale = 1
criterion = nn.BCELoss()

def to_img(x):

    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)

    return out
def generatelabels(batchsize,real_labels =None):
    x = torch.Tensor(torch.zeros(batchsize,10))#.to(device)
    if real_labels is None: #生成随机标签
        #y = [np.random.randint(0, 9) for i in range(batchsize)]
        y=[]
        for i in range(batchsize):
            tmp=np.random.randint(0,9)
            if(tmp==2):
                tmp=1
            y.append(tmp)
        x[np.arange(batchsize), y] = 1
    else:
        x[np.arange(batchsize),real_labels] = 1

    return x


if __name__ =="__main__":

    if not os.path.exists('./logires_img'):
        os.mkdir('./logires_img')

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))
    ])

    minist = datasets.MNIST(root='./data/',train = True,transform = img_transform,download=True)
    print("this is step")
    pca=PCA(n_components=3,copy=True)
    print("can get this")
    dataloader = torch.utils.data.DataLoader(
        dataset = minist,batch_size = batchsize,shuffle =True,
        drop_last = True
    )

    G1 = GANnet.generator(z_dimension,3136)
    G2= GANnet.generator(z_dimension,3136)

    #class_net_1=Net()
    class_net=LogisticRegression(784,10)
    G1.load_state_dict(torch.load("./generator_logi.pth"))
    G2.load_state_dict(torch.load("./generator.pth"))

    #class_net_1.load_state_dict(torch.load("./class.pth"))
    class_net.load_state_dict(torch.load("./class_logi.pth"))   
    #G.to(device)
    k=1
    #z = torch.Tensor(torch.randn(128, z_dimension)) #.to(device)
    #fig = plt.figure()
    a=[]
    b=[]
    ax = plt.subplot(111, projection='3d')
    for i, (imgs, real_labels) in enumerate(dataloader):
        imgs=imgs.view(-1,28*28)
        imgs.numpy()
        imgs=pca.fit_transform(imgs)
        label1_data=np.array(a)
        label2_data=np.array(a)
        for j in range(len(real_labels)):
            if(real_labels[j]==1):
                #ax.plot_surface(imgs[j][0],imgs[j][1],imgs[j][2],retride=1,csrede=1,cmap='yellow')
                ax.scatter(imgs[j][0],imgs[j][1],imgs[j][2], c='y')
                a.append(imgs[j].tolist())
                #label1_date=np.append()
            if(real_labels[j]==2):
                #ax.plot_surface(imgs[j][0],imgs[j][1],imgs[j][2],retride=1,csrede=1,cmap='red')
                ax.scatter(imgs[j][0],imgs[j][1],imgs[j][2], c='r')
                b.append(imgs[j].tolist())

    io.savemat('label1_date.mat',{'label1_date':a})
    io.savemat('label2_date.mat',{'label2_date':b})


    num_img =batchsize #imgs.size(0)
    z = torch.Tensor(torch.randn(num_img, z_dimension)) #.to(device)
    fake_labels = generatelabels(batchsize,1)
    fake_img = G1(z, fake_labels)
    fake_img=fake_img.view(-1,28*28)
        # print(fake_img.size())
    #imgs=imgs.view(-1,784)
    #imgs=imgs.numpy()
    #     print(imgs.shape)
    #pca.fit(imgs)
    #     print("sucees?")
    #imgs_t=pca.transform(imgs)
    #     print(imgs_t)
         
    output=class_net(fake_img)
    print(output)
                  
     
    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './logires_img/fake_images-{}.png'.format(i), nrow=16, padding=0)
    fake_img1=G2(z, fake_labels)
    fake_img1=fake_img1.view(-1,784).detach().numpy()
    fake_img1=pca.fit_transform(fake_img1)
    print(type(fake_img1))
    fake_img=fake_img.view(-1,784).detach().numpy()
    fake_img=pca.fit_transform(fake_img)
         #for i in range(128):
    ax.scatter(fake_img[:,0],fake_img[:,1],fake_img[:,2],marker='o',c='b')
    ax.scatter(fake_img1[:,0],fake_img1[:,1],fake_img1[:,2],marker='o',c='k')
    #plt.plot(fake_img[:,0],fake_img[:,1],color="red")
    #plt.plot(fake_img1[:,0],fake_img1[:,1],color="yellow")
    io.savemat('teacher.mat',{'label1_date':fake_img1})
    io.savemat('student.mat',{'label2_date':fake_img})


    plt.savefig('./logi_class.png',dpi=100)
    #D = GANnet.discriminator()

    #g_optimizer = torch.optim.Adam(G.parameters(),lr = 0.001)
    #d_optimizer = torch.optim.Adam(D.parameters(),lr = 0.001)

    #train_2(d_optimizer=d_optimizer,g_optimizer = g_optimizer,
    #      dataloader = dataloader, epoch_num = epoch_num,
    #      G=G,D=D,criterion = criterion)

