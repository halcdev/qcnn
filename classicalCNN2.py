import torch as tc
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import torch.multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('fork')

    class Cnn(nn.Module) :
        #using pooling instead of strides
        def __init__(self):
            super(Cnn,self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 3)  #3 input image channel, 6 output channels, 3x3 square convolution kernel
            self.pool1 = nn.MaxPool2d(2,2)  # Max pooling over a (2, 2) window
            self.conv2 = nn.Conv2d(6, 16, 3)  #6 input channels, 16 output channels, 3x3 square convolution kernel
            self.pool2 = nn.MaxPool2d(2,2)
            self.conv3 = nn.Conv2d(16, 32, 3)
            self.pool3 = nn.MaxPool2d(2,2)
            #Fully Connected layer if needed
            self.fc1 = nn.Linear(32 * 1 * 1, 120) 

        def forward(self, x):
            x = self.pool1(func.relu(self.conv1(x)))
            x = self.pool2(func.relu(self.conv2(x)))
            x = self.pool3(func.relu(self.conv3(x)))
            x = x.view(-1, 32 * 1 * 1)  # Flatten the tensor
            x = func.relu(self.fc1(x))
            return x
    
    #Training and Testing
    #Convert image data to tensor and normalize to have a zero mean and standard deviation of 1

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])  

    #Training the dataset
    trainset = datasets.MNIST(root = '/Users/pravinmahendran/Documents/GitHub/qcnn/', train = True, download = True, transform = transform)
    trainloader = tc.utils.data.DataLoader(trainset, batch_size = 5, shuffle = True, num_workers = 2)

    #Testing the dataset
    testset = datasets.MNIST(root = '/Users/pravinmahendran/Documents/GitHub/qcnn/', train = False,download = True,transform = transform  )
    testloader = tc.utils.data.DataLoader(testset, batch_size = 5, shuffle = True, num_workers = 2)

    device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

    model = Cnn()
    model = model.to(device)

    #specify the loss function to be used in the criterion variable
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum  = 0.9) #optimizer.SGD is the stochastic gradient descent

    #loop over the dataset to train
    for epoch in range(5) : 
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad() 
            outputs = model(inputs)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print("Training Complete")

    #Check the accuracy
    correct = 0
    total = 0
    with tc.no_grad() :
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = tc.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy is %d %%'% (100 * correct/total))








