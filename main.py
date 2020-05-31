import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

# convert data to torch.FloatTensor
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#transform = transforms.ToTensor()

train_data = datasets.ImageFolder(root = 'food_stitched_semisupervised', transform = transform)

# Create training and test dataloaders

num_workers = 0
# how many samples per batch to load
batch_size = 20

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle = True)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        hdim = 16
        bdim = 16
        self.conv1 = nn.Conv2d(3, hdim, 3, padding=1)
        self.conv2 = nn.Conv2d(hdim, hdim, 3, padding=1)
        self.conv3 = nn.Conv2d(hdim, bdim, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.convB = nn.Conv2d(bdim, 3, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.actvn = nn.PReLU()
        
    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        l = x[:,:,:,:128]
        m = x[:,:,:,128:256]
        r = x[:,:,:,256:]
        
        l = self.actvn(self.conv1(l))
        l = self.pool(l)
        l = self.actvn(self.conv2(l))
        l = self.pool(l)
        l = self.actvn(self.conv3(l))
        l = self.convB(l)
        
        m = self.actvn(self.conv1(m))
        m = self.pool(m)
        m = self.actvn(self.conv2(m))
        m = self.pool(m)
        m = self.actvn(self.conv3(m))
        m = self.convB(m)
        
        r = self.actvn(self.conv1(r))
        r = self.pool(r)
        r = self.actvn(self.conv2(r))
        r = self.pool(r)
        r = self.actvn(self.conv3(r))
        r = self.convB(r)
        
        l = torch.flatten(l, start_dim=1)
        m = torch.flatten(m, start_dim=1)
        r = torch.flatten(r, start_dim=1)

        return l, m, r

class TripletAlexNet(nn.Module):
    def __init__(self):
        super(TripletAlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2), #60 / 59(?)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4, stride = 1), #57
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=5, stride = 2), # 27
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1), # 14
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fcn = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(96*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(4096, 4096)
        )
        

    def forward(self, x):
        l = x[:,:,:,:128]
        l = l + ((torch.ones(l.size()))*(torch.rand((1))-0.5)*0.3).to('cuda')
        l[l>1.0] = 1.0
        l[l<-1.0] = -1.0
        
        m = x[:,:,:,128:256]
        m = m + ((torch.ones(m.size()))*(torch.rand((1))-0.5)*0.3).to('cuda')
        m[m>1.0] = 1.0
        m[m<-1.0] = -1.0
        
        r = x[:,:,:,256:]
        r = r + ((torch.ones(r.size()))*(torch.rand((1))-0.5)*0.3).to('cuda')
        r[r>1.0] = 1.0
        r[r<-1.0] = -1.0
        
        # Alex-net
        L = self.features(l)
        L = torch.flatten(L, 1)
        L = self.fcn(L)
        L = F.normalize(L,dim=1,p=2)
        
        M = self.features(m)
        M = torch.flatten(M, 1)
        M = self.fcn(M)
        M = F.normalize(M,dim=1,p=2)
        
        R = self.features(r)
        R = torch.flatten(R, 1)
        R = self.fcn(R)
        R = F.normalize(R,dim=1,p=2)
        
        return L, M, R

# Initialize model    
#model = TripletNetwork()
model = TripletAlexNet()
model.cuda()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)

# number of epochs to train the model
n_epochs = 10000
ep = 1
for epoch in range(1, n_epochs+1):
    
    # monitor training loss
        
    ###################
    # train the model #
    ###################
    it = 0
    train_loss = 0.0
    for data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
        images, _ = data
        images = images.to('cuda')
        optimizer.zero_grad()
        l, m, r = model(images)

        # loss
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        loss = triplet_loss(l,m,r)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)
        
        it = it + 1
        
        if it%200 == 0:
            print("Iteration: {} Loss: {}".format(it,100*loss))

        if it%1000 == 0:
            #print('Saving model')
            torch.save(model.state_dict(), "/content/drive/My Drive/IML/task4/out_alexnet_paper/anpaper_ss.pt")
          
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        100*train_loss
        ))
    
    print('Saving model')
    torch.save(model.state_dict(), "/content/drive/My Drive/IML/task4/out_alexnet_paper/anpaper_ss_epoch{}.pt".format(ep))
    ep = ep + 1
