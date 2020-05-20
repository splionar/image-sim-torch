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
batch_size = 40

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        
        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride = 2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, 4, stride = 2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 128, 4, stride = 2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 256, 4, stride = 2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 512, 4, stride = 2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        #Bottleneck
        self.conv6 = nn.Conv2d(512, 4000, 4, stride = 1)
        self.dropout = nn.Dropout(p=0.25, inplace=False)
        
        ## decoder layers ##
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(4000, 512, 4, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        # Activation
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.encode(x)

        x = self.conv6(x)
        x = self.leakyRelu(x)
        x = self.dropout(x)

        x = self.decode(x)
     
        return x

# initialize the NN
model = ConvAutoencoder()

# specify loss function
criterion = nn.MSELoss()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 100

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
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images[:,:,:,:128])
        # calculate the loss
        loss = 0.6*criterion(outputs, images[:,:,:,128:256]) / 0.4*criterion(outputs, images[:,:,:,256:])
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
            torch.save(model.state_dict(), "/content/drive/My Drive/IML/task4/weights_40k_div.pt")
            
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        100*train_loss
        ))
    
    print('Backup model')
    torch.save(model.state_dict(), "/content/drive/My Drive/IML/task4/weights_epoch_40k_div.pt")
