import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

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

        div = 4

        self.features = nn.Sequential(
            nn.Conv2d(3, int(64/div), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(int(64/div), int(192/div), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(int(192/div), int(384/div), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(int(384/div), int(256/div), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(256/div), int(256/div), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveMaxPool2d((8, 8))
        self.fcn = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(int(256/div) * 8 * 8, int(4096/2)),
        )
        
        #Multi-scale
        hdim = 16
        bdim = 16
        
        #2:1 subsample
        self.subsample1_cnn = nn.Sequential(
            nn.MaxPool2d((2,2)),
            nn.Conv2d(3, hdim, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(hdim, hdim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hdim, bdim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(bdim, 3, 3, padding=1)    
            )
         
        self.subsample1_fcn = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(3072, int(4096/div*2))
                )
        #4:1 subsample
        self.subsample2_cnn = nn.Sequential(
            nn.MaxPool2d((2,2)),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(3, hdim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hdim, hdim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hdim, bdim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(bdim, 3, 3, padding=1)    
            )        

        self.subsample2_fcn = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(3072, int(4096/div*2))
                )

    def forward(self, x):
        l = x[:,:,:,:128]
        l = l + ((torch.ones(l.size()))*(torch.rand((1))-0.5)*0.3).to('cuda')
        l[l>1.0] = 1.0
        l[l<0.0] = 0.0
        
        m = x[:,:,:,128:256]
        m = m + ((torch.ones(m.size()))*(torch.rand((1))-0.5)*0.3).to('cuda')
        m[m>1.0] = 1.0
        m[m<0.0] = 0.0
        
        r = x[:,:,:,256:]
        r = r + ((torch.ones(r.size()))*(torch.rand((1))-0.5)*0.3).to('cuda')
        r[r>1.0] = 1.0
        r[r<0.0] = 0.0
        
        # Alex-net
        L = self.features(l)
        L = self.avgpool(L)
        L = torch.flatten(L, 1)
        L = self.fcn(L)
        
        M = self.features(m)
        M = self.avgpool(M)
        M = torch.flatten(M, 1)
        M = self.fcn(M)
        
        R = self.features(r)
        R = self.avgpool(R)
        R = torch.flatten(R, 1)
        R = self.fcn(R)
        
        # Subsample 1 2:1 + shallow CNN
        L_sub1 = self.subsample1_cnn(l) 
        L_sub1 = torch.flatten(L_sub1, 1)
        L_sub1 = self.subsample1_fcn(L_sub1)
        
        M_sub1 = self.subsample1_cnn(m) 
        M_sub1 = torch.flatten(M_sub1, 1)
        M_sub1 = self.subsample1_fcn(M_sub1)
        
        R_sub1 = self.subsample1_cnn(r) 
        R_sub1 = torch.flatten(R_sub1, 1)
        R_sub1 = self.subsample1_fcn(R_sub1)
        
        # Subsample 2 4:1 + shallow CNN
        L_sub2 = self.subsample2_cnn(l) 
        L_sub2 = torch.flatten(L_sub2, 1)
        L_sub2 = self.subsample2_fcn(L_sub2)
        
        M_sub2 = self.subsample2_cnn(m) 
        M_sub2 = torch.flatten(M_sub2, 1)
        M_sub2 = self.subsample2_fcn(M_sub2)
        
        R_sub2 = self.subsample2_cnn(r) 
        R_sub2 = torch.flatten(R_sub2, 1)
        R_sub2 = self.subsample2_fcn(R_sub2)
        
        L = L + L_sub1 + L_sub2
        M = M + M_sub1 + M_sub2
        R = R + R_sub1 + R_sub2
        
        return L, M, R

# Initialize model    
model = TripletNetwork()
#model = TripletAlexNet()
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
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        l, m, r = model(images)

        # loss
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        loss = triplet_loss(l,m,r)
        #l2_plus = torch.mean(torch.square(l-m),dim=1) # size = batch_size,
        #l2_min = torch.mean(torch.square(l-r),dim=1) # size = batch_size,
        #loss = torch.mean(F.relu(l2_plus - l2_min + 0.8))

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
        
        it = it + 1
        
        if it%200 == 0:
            print("Iteration: {} Loss: {}".format(it,100*loss))

        if it%1000 == 0:
            #print('Saving model')
            torch.save(model.state_dict(), "/content/drive/My Drive/IML/task4/out_mulscale/mulscale_ss.pt")
          
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        100*train_loss
        ))
    
    print('Saving model')
    torch.save(model.state_dict(), "/content/drive/My Drive/IML/task4/out_mulscale/mulscale_ss_epoch{}.pt".format(ep))
    ep = ep + 1
