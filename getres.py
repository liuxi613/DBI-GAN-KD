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
            tmp=np.random.randint(1,2)
            '''
            if(tmp==2):
                tmp=1
            '''
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
    pca=PCA(n_components=2,copy=True)
    print("can get this")
    dataloader = torch.utils.data.DataLoader(
        dataset = minist,batch_size = batchsize,shuffle =True,
        drop_last = True
    )

    #G1 = GANnet.generator(z_dimension,3136)
    G2= GANnet.generator(z_dimension,3136)

    class_net=Net()
    #class_net=LogisticRegression(784,10)
    #G1.load_state_dict(torch.load("./generator_logi.pth"))
    G2.load_state_dict(torch.load("./yuanshi_generator.pth"))

    class_net.load_state_dict(torch.load("./class.pth"))
    #class_net.load_state_dict(torch.load("./class_logi.pth"))   
    #G.to(device)
    k=1
    #z = torch.Tensor(torch.randn(128, z_dimension)) #.to(device)
    
    for i, (imgs, real_labels) in enumerate(dataloader):
         num_img = imgs.size(0)
         z = torch.Tensor(torch.randn(num_img, z_dimension)) #.to(device)
         fake_labels = generatelabels(batchsize)
         fake_img = G2(z, fake_labels)
         #fake_img=fake_img.view(-1,28*28)
         #print(fake_img.size())
         #imgs=imgs.view(-1,784)
         #imgs=imgs.numpy()
        # print(imgs.shape)
         #pca.fit(imgs)
         #print("sucees?")
         #imgs_t=pca.transform(imgs)
         '''
         print(imgs_t)
         for i in range(len(real_labels)):
             if(real_labels[i]==1):
                 plt.scatter(imgs[i][0],imgs[i][1],marker='o',c='b')
             if(real_labels[i]==2):
                 plt.scatter(imgs[i][0],imgs[i][1],marker='o',c='y')
         '''
         output=class_net(fake_img)
         '''
         for i in range(len(output)):
             key=torch.max(output[i],dim=0)[1]
             #if ((output[i][1]>0.1 or output[i][2]>0.1) and abs(output[i][1]-output[i][2])<0.1):
             if ((key==1 or key==2) and abs(output[i][1]-output[i][2])<0.1 and (output[i][1]>0.45 and output[i][1]<0.55 and output[i][2]>0.45 and output[i][2]<0.55)):
                 fake_images = to_img(fake_img[i].cpu().data)
                 save_image(fake_images, './res_img/fake_images_1-{}.png'.format(k), padding=0)
                  
                 print(output[i])
                 k=k+1
         #print(output.data.max(1, keepdim=True)[1])
         '''
         for i in range(len(output)):      
              fake_images = to_img(fake_img[i].cpu().data)
              save_image(fake_images, './res_img/yuanshi_fake_images-{}.png'.format(k), padding=0)
              k=k+1
         '''
         fake_img1=G2(z, fake_labels)
         fake_img1=fake_img1.view(-1,784).detach().numpy()
         fake_img1=pca.transform(fake_img1)
         print(type(fake_img1))
         fake_img=fake_img.view(-1,784).detach().numpy()
         fake_img=pca.transform(fake_img)
         #for i in range(128):
         plt.scatter(fake_img[:,0],fake_img[:,1],marker='o',c='r')
         plt.scatter(fake_img1[:,0],fake_img1[:,1],marker='o',c='k')
         plt.savefig('./logi_class.png',dpi=100)
         '''
    #D = GANnet.discriminator()
'''
    #g_optimizer = torch.optim.Adam(G.parameters(),lr = 0.001)
    #d_optimizer = torch.optim.Adam(D.parameters(),lr = 0.001)

    #train_2(d_optimizer=d_optimizer,g_optimizer = g_optimizer,
    #      dataloader = dataloader, epoch_num = epoch_num,
    #      G=G,D=D,criterion = criterion)
'''
