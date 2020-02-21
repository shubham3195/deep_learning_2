import numpy as np
import torch
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
                                                              
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


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


class CNN(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        

nnn = Classifier()
nnn.load_state_dict(torch.load("./models/MLP.pt"))

nnn.eval()
f1=open("multi-layer-net.txt","w")
 
#nn.eval()
count=0
j=0
yp=[]
yt=[]

n1=nn.NLLLoss()
n2=nn.CrossEntropyLoss()

rl=0
ls=0

for images, labels in testloader:
       lo = nnn(images)
       l=lo.max(1)[1]
       ls=n1(lo,labels)
       rl=rl+ls.item()
       for i in range(len(images)):
            if(l[i]==labels[i]):
                count=count+1
            yp.append(l[i])
            yt.append(labels[i])
            #f1.write(str(l[i]))
            #f1.write("\n")


f1.write("Loss on Test Data : ")
f1.write(str(rl/len(testloader)))
f1.write("\n")
f1.write("Accuracy on Test Data : ")
f1.write(str(count/len(testset)))
f1.write("\n")
f1.write("gt_label,pred_label \n")

for i in range(len(yp)):
       f1.write(str(np.array(yt[i]))+",")
       f1.write(str(np.array(yp[i])))
       f1.write("\n")


        
print("Accuracy for MLP : ",count/len(testset))
print(" ")
print("Confusion Matrix for MLP ")
print(" ")
from sklearn import metrics
confusion_m= metrics.confusion_matrix(yp,yt)
print(confusion_m)


cnn = CNN()
cnn.load_state_dict(torch.load('./models/convNet.pt'))
f2=open("convolution-neural-net.txt","w")

count=0
j=0
yp=[]
yt=[]

rl=0
ls=0

for images, labels in testloader:
        lo = cnn(images)
        l=lo.max(1)[1]

        ls=n2(lo,labels)
        rl=rl+ls.item()
        for i in range(len(images)):
            if(l[i]==labels[i]):
                count=count+1
            yp.append(l[i])
            yt.append(labels[i])
            #f2.write(str(l[i]))
            #f2.write("\n")

f2.write("Loss on Test Data : ")
f2.write(str(rl/len(testloader)))
f2.write("\n")
f2.write("Accuracy on Test Data : ")
f2.write(str(count/len(testset)))
f2.write("\n")
f2.write("gt_label,pred_label \n")

for i in range(len(yp)):
       f2.write(str(np.array(yt[i]))+",")
       f2.write(str(np.array(yp[i])))
       f2.write("\n")


        
print("Accuracy for CNN : ",count/len(testset))
print(" ")
print("Confusion Matrix for CNN ")
print(" ")
confusion_m= metrics.confusion_matrix(yp,yt)
print(confusion_m)                            
