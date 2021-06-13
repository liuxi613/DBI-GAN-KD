import torch
import  torch.nn as nn
import torch.functional as F
import os
from tensorboardX import SummaryWriter
import torchvision
from torchvision import datasets
#from getministac import Net
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image,make_grid
import cGANnets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#超参数
batchsize = 128
z_dimension = 100
num = 25
epoch_num  = 12
scale = 1
criterion = nn.BCELoss()

writer = SummaryWriter()
def train_1(d_optimizer, g_optimizer, dataloader, epoch_num, G, D,C, criterion):
    '''
    #这个策略训练失败
    :param d_optimizer:
    :param g_optimizer:
    :param dataloader:
    :param epoch_num:
    :param G:
    :param D:
    :param criterion:
    :return:
    '''
    G.to(device)
    D.to(device)
    # g_optimizer.to(device)
    # d_optimizer.to(device)
    # criterion.to(device)
    step = 0
    for epoch in range(epoch_num):
        for i, (imgs, real_labels) in enumerate(dataloader):
            num_img = imgs.size(0)
            real_label = torch.Tensor(torch.ones(num_img)).to(device)
            fake_label = torch.Tensor(torch.zeros(num_img)).to(device)

            real_labels = generatelabels(batchsize, real_labels)  # 产生对应的one-hot编码标签
            real_labels.requires_grad = True

            imgs = imgs.to(device)
            num_img = imgs.size(0)

            real_out = D(imgs, real_labels)  # 输入真实图片得到结果
            real_scores = real_out
            d_loss_real = criterion(real_out, real_label)

            z = torch.Tensor(torch.randn(num_img, z_dimension)).to(device)

            fake_labels = generatelabels(batchsize)  # 生成编码标签
            fake_labels.requires_grad = True

            z.requires_grad = True
            fake_img = G(z, fake_labels)
            fake_out = D(fake_img, fake_labels)
            '''
            for i in range(fake_labels):
                if (fake_labels[i][1]==1 or fake_labels[i][2]==1):
                    fake_labels[i][1]=0.5
                    fake_labels[i][2]=0.5
            '''
            c_out=C(fake_img)
            c_lossf = nn.MSELoss()
            
            c_loss=c_lossf(c_out,fake_labels)
            d_loss_fake = criterion(fake_out, fake_label)
            fake_scores = fake_out  #

            # 先更新判别器参数 然后再更新生成器参数
            d_loss = d_loss_real + d_loss_fake
            writer.add_scalar('d_loss', scale * d_loss, step)

            # 第一个epoch先充分训练判别器 所以每十次迭代才更新一次生成器
            if epoch == 0:
                d_optimizer.zero_grad()  # 梯度清零
                d_loss.backward()  # 计算梯度
                d_optimizer.step()  # 更新参数

                # 更新生成器
                if i % 10 == 0:
                    z = torch.Tensor(torch.randn(num_img, z_dimension)).to(device)
                    z.requires_grad = True
                    fake_img = G(z, fake_labels)
                    fake_out = D(fake_img, fake_labels)
                    g_loss = criterion(fake_out, real_label)
                    writer.add_scalar('g_loss', scale * g_loss, step)

                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()

            else:  # 后面的迭代每隔25次迭代才更新一次判别器
                if i % num == 0:
                    d_optimizer.zero_grad()  # 梯度清零
                    d_loss.backward()  # 计算梯度
                    d_optimizer.step()  # 更新参数

                # 更新生成器
                z = torch.Tensor(torch.randn(num_img, z_dimension)).to(device)
                z.requires_grad = True
                fake_labels = generatelabels(batchsize)
                fake_labels.requires_grad = True

                fake_img = G(z, fake_labels)
                fake_out = D(fake_img, fake_labels)
                g_loss = criterion(fake_out, real_label)+c_loss

                writer.add_scalar('g_loss', scale * g_loss, step)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            if (i + 1) % 50 == 0:
                print('Epoch[{}/{}],d_loss: {:.6f},g_loss: {:.6f}'
                      'D real: {:.6f}, D fake: {:.6f}'.format(
                    epoch, epoch_num, d_loss * scale, g_loss * scale,
                    real_scores.data.mean(), fake_scores.data.mean()
                )
                )

            step += 1

        if epoch == 0:
            real_images = to_img(imgs.cpu().data)

            save_image(real_images, './img/real_images.png', nrow=16, padding=0)
        fake_images = to_img(fake_img.cpu().data)

        grid = make_grid(fake_images, nrow=16, padding=0)
        writer.add_image('image', grid, epoch)

        save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1), nrow=16, padding=0)

    # 训练完成后保存模型文件
    torch.save(G.state_dict(), './generator.pth')
    torch.save(D.state_dict(), './discriminator.pth')
def loss(criterion,c_lossf,c_out,fake_labels,fake_out,real_label,k,soft_re):
    re,re1=soft_re.sort(1,descending=True)
    suml=0
    for i in range(len(soft_re)):
         suml+=(soft_re[i][re1[i][0]]-soft_re[i][re1[i][1]])
    if k<=20:
         genloss= criterion(fake_out, real_label)
    else:
        genloss=0
    c_loss = c_lossf(c_out,fake_labels)
    print(genloss,c_loss,suml/len(soft_re))
    return 0.5*genloss+c_loss+suml/len(soft_re)

    


def train_2(d_optimizer, g_optimizer, dataloader, epoch_num, G, D,C, criterion):
    '''

    :param d_optimizer:
    :param g_optimizer:
    :param dataloader:
    :param epoch_num:
    :param G:
    :param D:
    :param criterion:
    :return:
    '''
    G.to(device)
    D.to(device)
    C.to(device)
    # g_optimizer.to(device)
    # d_optimizer.to(device)
    # criterion.to(device)
    step = 0
    for epoch in range(epoch_num):
        for i, (imgs, real_labels) in enumerate(dataloader):
            num_img = imgs.size(0)
            real_label = torch.Tensor(torch.ones(num_img)).to(device)
            fake_label = torch.Tensor(torch.zeros(num_img)).to(device)
            #print(real_labels)
            real_labels = real_labels.view(-1,1,1).long()#.view(1,-1)[0].long()
            #print("real_labels",real_labels)
            record_real=real_labels
            for j in range(0,len(real_labels)):
                #print(real_labels[i])
 
                #if (real_labels[j]==1):
                real_labels[j]=0
            #print(real_labels)
            
            real_labels = generatelabels(num_img, real_labels)  # 产生对应的one-hot编码标签
            
            real_labels.requires_grad = True

            imgs = imgs.to(device)
            num_img = imgs.size(0)
            #print(imgs.shape)
            real_out = D(imgs, real_labels)  # 输入真实图片得到结果
            real_scores = real_out
            d_loss_real = criterion(real_out, real_label)

            z = torch.Tensor(torch.randn(num_img, z_dimension)).to(device)

            fake_labels = generatelabels(num_img,0) # 生成编码标签
            fake_labels.requires_grad = True

            z.requires_grad = True
            fake_img = G(z, fake_labels)
            fake_out = D(fake_img, fake_labels)
            ssss=nn.functional.softmax(C(fake_img),1)
            rere,rere1=ssss.sort(1,descending=True)
            lossp=(rere[:,0]-rere[:,1]).sum()/num_img
            print(rere[:,0],rere[:,1])
            d_loss_fake = criterion(fake_out, fake_label)

            fake_scores = fake_out  #
            #for i in range(fake_labels):
            #    if (fake_labels[i][1]==1 or fake_labels[i][2]==1):
            #        fake_labels[i][1]=0.5
            #        fake_labels[i][2]=0.5
            #c_out=C(fake_img)
            #c_lossf = nn.MSELoss()

            #c_loss=c_lossf(c_out,fake_labels)
            #d_loss_fake = criterion(fake_out, fake_label)
            print("kk",d_loss_real,d_loss_fake,lossp)

            # 先更新判别器参数 然后再更新生成器参数
            d_loss = d_loss_real + d_loss_fake+lossp
            writer.add_scalar('d_loss', scale * d_loss, step)

            # 第一个epoch先充分训练判别器 所以每十次迭代才更新一次生成器
            # if epoch == 0:
            d_optimizer.zero_grad()  # 梯度清零
            d_loss.backward()  # 计算梯度
            d_optimizer.step()  # 更新参数

                # # 更新生成器
                # if i % 10 == 0:
            for k in range(1):
                z = torch.Tensor(torch.randn(num_img, z_dimension)).to(device)
                z.requires_grad = True
                fake_img = G(z, fake_labels)
                fake_out = D(fake_img, fake_labels)
            #print(fake_labels)
            #print(type(fake_labels))
            #print(fake_labels)
            #fake_img = fake_img.view(-1, 28 * 28)
                c_out=C(fake_img)
                ss=nn.functional.softmax(c_out,1)
                b,b1=ss.sort(1,descending=True)
                print(ss.shape)
                with torch.no_grad():
                    for i in range(num_img):
                        #if (fake_labels[i][1]==1):
                        fake_labels[i][0]=0
                        fake_labels[i][b1[0]]=0
                        #for j in range(len(fake_labels[i])):
                    #    if a!=fake_labels[i].argmax(dim=1):
                            
                        fake_labels[i][b1[1]]=1

                print(nn.functional.softmax(c_out,1))
                c_lossf = nn.MSELoss()
                #c_loss=c_lossf(c_out,fake_labels)
                #c_loss.requires_grad=False
                g_loss = criterion(fake_out, real_label)
                g_loss=loss(criterion,c_lossf,nn.functional.softmax(c_out,1),fake_labels,fake_out,real_label,epoch,ss)
                print(g_loss)
                writer.add_scalar('g_loss', scale * g_loss, step)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            # else:  # 后面的迭代每隔25次:迭代才更新一次判别器
            #     if i % num == 0:
            #         d_optimizer.zero_grad()  # 梯度清零
            #         d_loss.backward()  # 计算梯度
            #         d_optimizer.step()  # 更新参数
            #
            #     # 更新生成器
            #     z = torch.Tensor(torch.randn(num_img, z_dimension)).to(device)
            #     z.requires_grad = True
            #     fake_labels = generatelabels(batchsize)
            #     fake_labels.requires_grad = True
            #
            #     fake_img = G(z, fake_labels)
            #     fake_out = D(fake_img, fake_labels)
            #     g_loss = criterion(fake_out, real_label)
            #
            #     writer.add_scalar('g_loss', scale * g_loss, step)
            #
            #     g_optimizer.zero_grad()
            #     g_loss.backward()
            #     g_optimizer.step()

            if (i + 1) % 50 == 0:
                print('Epoch[{}/{}],d_loss: {:.6f},g_loss: {:.6f}'
                      'D real: {:.6f}, D fake: {:.6f}'.format(
                    epoch, epoch_num, d_loss * scale, g_loss * scale,
                    real_scores.data.mean(), fake_scores.data.mean()
                )
                )

            step += 1

        if epoch == 0:
            real_images = to_img(imgs.cpu().data)

            save_image(real_images, './img/real_images.png', nrow=16, padding=0)
        fake_images = to_img(fake_img.cpu().data)

        grid = make_grid(fake_images, nrow=16, padding=0)
        writer.add_image('image', grid, epoch)

        save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1), nrow=16, padding=0)

    # 训练完成后保存模型文件
    torch.save(G.state_dict(), './yuanshi_generator.pth')
    torch.save(D.state_dict(), './discriminator.pth')

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


def to_img(x):

    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 3, 32, 32)

    return out

