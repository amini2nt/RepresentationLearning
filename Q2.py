import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
import torchvision.transforms



class Classifier(nn.Module):
    """Convnet Classifier"""
    def layer(self, i, o):
        return [nn.Conv2d(in_channels=i, out_channels=o, kernel_size=(3, 3), padding=1),
                nn.Dropout(p=0.3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)]

    def __init__(self):
        super(Classifier, self).__init__()
        layers = self.layer(1,16)+self.layer(16,32)+self.layer(32,64)+self.layer(64,128)
        self.conv = nn.Sequential(*layers)
        self.clf = nn.Linear(128, 10)

    def forward(self, x):
        return self.clf(self.conv(x).squeeze())

def main():
    mnist_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=mnist_transforms, download=True)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=mnist_transforms, download=True)

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True, num_workers=2)

    cuda_available = torch.cuda.is_available()
    print("Cuda available: %s" % cuda_available)

    clf = Classifier()
    if cuda_available:
        clf = clf.cuda()
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() #  LogSoftmax and NLLLoss

    for epoch in range(10):
        losses = []
        # Train
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = clf(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())
            
            if batch_idx%50==0:
                print('Epoch : %d Loss : %.3f ' % (epoch, np.mean(losses)))
        
        # Evaluate
        clf.eval()
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = clf(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print('Epoch : %d Test Acc : %.2f, number of tests : %d' % (epoch, 100*float(correct)/total, total))
        print('--------------------------------------------------------------')
        clf.train()

if __name__ == "__main__":
    main()