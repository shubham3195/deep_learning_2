import numpy as np
import torch
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

image, label = next(iter(trainloader))


print("Department : Computer Science and Automation(CSA)")
print("Name : shubham sharma")
print("Sr.No. : 16013")

# neural net for our task
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 150
tl=[]
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
    	z=running_loss/len(trainloader)
    	print("Training loss:",z)
    	tl.append(z)
#plt.plot(tl)
#plt.show()

count=0
j=0
yp=[]
yt=[]
for images, labels in testloader:
        lo = model(images)
        l=lo.max(1)[1]
        for i in range(len(images)):
            if(l[i]==labels[i]):
                count=count+1
            yp.append(l[i])
            yt.append(labels[i])
        
print("Accuracy : ",count/len(testset))


from sklearn import metrics
confusion_m= metrics.confusion_matrix(yp,yt)
print(confusion_m)


fig = plt.figure()
x=plt.plot(tl[0:80])

plt.title('training-loss-mlp ')
fig.savefig("/img/training_mlp.png")

torch.save(model.state_dict(), "/models/MLP.pt")

