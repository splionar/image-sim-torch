import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

train_data = datasets.ImageFolder(root = 'food_stitched_40k', transform = transform)

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

# Initialize model    
model = TripletNetwork()
model.cuda()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 10000

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
        l2_plus = torch.mean(torch.square(l-m),dim=1) # size = batch_size,
        l2_min = torch.mean(torch.square(l-r),dim=1) # size = batch_size,
        loss = torch.mean(F.relu(l2_plus - l2_min + 0.2))

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
        
        it = it + 1
        
        if it%20 == 0:
            print("Iteration: {} Loss: {}".format(it,100*loss))

        if it%100 == 0:
            #print('Saving model')
            torch.save(model.state_dict(), "/content/drive/My Drive/IML/task4/triplet_40k_colab.pt")
          
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        100*train_loss
        ))
    
    print('Saving model')
    torch.save(model.state_dict(), "/content/drive/My Drive/IML/task4/triplet_40k_colab_epoch.pt")
