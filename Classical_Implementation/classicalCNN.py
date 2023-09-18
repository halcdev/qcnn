import torch.nn as nn

class mnistCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2), #B&W image, with 16 different 5x5 sliding window -> output depth of 16
            nn.ReLU(), #Non-linear activation functions; see sigmoid, tanh, leaky ReLU etc.
            nn.Conv2d(16, 32, 3, 2, 1), #Change parameters, add more conv layers...
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 0),
            nn.ReLU()
            #nn.MaxPool2d(kernel_size=3)
        )
        #Calculate linear layer to equal final convolution output total size
        self.out = nn.Linear(32 * 12 * 12, 10)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) #Collapse into a 2D matrix
        output = self.out(x)
        return output