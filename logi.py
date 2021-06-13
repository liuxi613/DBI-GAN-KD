import torch
import torch.nn.functional as F
import numpy as np 
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as Data
batch_size = 128
BATCH_SIZE=256
n_iters = 1000
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



#minist = datasets.MNIST(root='./data/',train = True,transform = img_transform,download=True)
minist = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)


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


epochs = n_iters / (len(train_loader) / BATCH_SIZE)
input_dim = 3072
output_dim = 2
lr_rate = 0.01
model = seaNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
iter = 0

for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)#.view(-1, 3*32 * 32))
        labels = Variable(labels).view(1,-1)[0]
        #print(labels)
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        iter+=1
        if iter%500==0:
            # 计算准确率
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images)#.view(-1, 3*32*32))
                outputs = model(images)
                #print(labels)
                labels = Variable(labels).view(1,-1)[0].long()
                _, predicted = torch.max(outputs.data, 1)
                #print(predicted,labels)
                total+= labels.size(0)
                # 如果用的是 GPU，则要把预测值和标签都取回 CPU，才能用 Python 来计算
                correct+= (predicted == labels).sum()
            accuracy = 100. * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))


torch.save(model.state_dict(), './class_logi.pth')
