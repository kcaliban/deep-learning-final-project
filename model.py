import torch.nn as nn
import torch

class ConvTransposeNet(nn.Module):
    def __init__(self, kernel, num_in_channels, num_filters, device):
        super().__init__()

        # Useful parameters
        stride = 1
        padding = 0

        # First Group
        self.group1 = nn.Sequential(
            nn.Conv2d(num_in_channels, 4 * num_filters, kernel_size=kernel, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=4*num_filters,device=device),
            nn.ReLU(),
        )

        # Second Group
        self.group2 = nn.Sequential(
            nn.Conv2d(4 * num_filters, 2 * num_filters, kernel_size=kernel, stride=stride, padding=padding, device=device),
            nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=2*num_filters, device=device),
            nn.ReLU(),
        )

        # Linear layer
        self.linear = nn.Linear(64, 1, device=device)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.group1(x)
        x = self.group2(x)
        # x = self.group3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.sigmoid(x/100)
        return x
    
class CustomNet(nn.Module):
    def __init__(self, device):
        super().__init__()

        stride = 1
        padding = 0

        self.group1 = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=padding),
            nn.BatchNorm2d(num_features=32,device=device),
            nn.ReLU(),
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=8, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=32, device=device),
            nn.ReLU(),
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=2, stride=stride, padding=padding, device=device),
            nn.MaxPool2d(kernel_size=2, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=16, device=device),
            nn.ReLU(),
        )

        # Linear layer
        self.linear = nn.Linear(16 + 32, 1, device=device)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       x1 = self.group1(x)
       # print(x1.size())
       x2 = self.group2(x)
       # print(x2.size())
       x3 = self.group3(x1)
       # print(x3.size())
       x4 = torch.cat((x2, x3), dim=1)
       x5 = torch.flatten(x4, start_dim=1)
       x6 = self.linear(x5)
       return self.sigmoid(x6/100)
    

class CustomNetV2(nn.Module):
    def __init__(self, device):
        super().__init__()

        stride = 1
        padding = 0

        self.group1 = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=padding),
            nn.BatchNorm2d(num_features=32,device=device),
            nn.ReLU(),
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=8, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=32, device=device),
            nn.ReLU(),
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=2, stride=stride, padding=padding, device=device),
            nn.MaxPool2d(kernel_size=2, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=16, device=device),
            nn.ReLU(),
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=1, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=16, device=device),
            nn.ReLU(),
        )

        # Linear layer
        self.linear = nn.Linear(16, 1, device=device)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       x1 = self.group1(x)
       # print(x1.size())
       x2 = self.group2(x)
       # print(x2.size())
       x3 = self.group3(x1)
       # print(x3.size())
       x4 = torch.cat((x2, x3), dim=1)
       x5 = self.group4(x4)
       x6 = torch.flatten(x5, start_dim=1)
       x7 = self.linear(x6)
       return self.sigmoid(x7/100)


class CustomNetV2Div1000(nn.Module):
    def __init__(self, device):
        super().__init__()

        stride = 1
        padding = 0

        self.group1 = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=padding),
            nn.BatchNorm2d(num_features=32,device=device),
            nn.ReLU(),
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=8, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=32, device=device),
            nn.ReLU(),
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=2, stride=stride, padding=padding, device=device),
            nn.MaxPool2d(kernel_size=2, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=16, device=device),
            nn.ReLU(),
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=1, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=16, device=device),
            nn.ReLU(),
        )

        # Linear layer
        self.linear = nn.Linear(16, 1, device=device)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       x1 = self.group1(x)
       # print(x1.size())
       x2 = self.group2(x)
       # print(x2.size())
       x3 = self.group3(x1)
       # print(x3.size())
       x4 = torch.cat((x2, x3), dim=1)
       x5 = self.group4(x4)
       x6 = torch.flatten(x5, start_dim=1)
       x7 = self.linear(x6)
       return self.sigmoid(x7/1000)

# DIV 1000!
class CustomNetV3(nn.Module):
    def __init__(self, device):
        super().__init__()

        stride = 1
        padding = 0

        self.group1 = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=32,device=device),
            nn.ReLU(),
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=8, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=32, device=device),
            nn.ReLU(),
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=16, device=device),
            nn.ReLU(),
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=padding),
            nn.BatchNorm2d(num_features=8, device=device),
            nn.ReLU(),
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(32 + 8, 16, kernel_size=1, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=16, device=device),
            nn.ReLU(),
        )

        # Linear layer
        self.linear = nn.Linear(16, 1, device=device)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       x1 = self.group1(x)
       # print(x1.size())
       x2 = self.group2(x)
       # print(x2.size())
       x3 = self.group3(x1)
       x4 = self.group4(x3)
       # print(x3.size())
       x5 = torch.cat((x2, x4), dim=1)
       x6 = self.group5(x5)
       x7 = torch.flatten(x6, start_dim=1)
       x8 = self.linear(x7)
       return self.sigmoid(x8/1000)


# DIV 1000!
class CustomNetV3MoreFeatures(nn.Module):
    def __init__(self, device):
        super().__init__()

        stride = 1
        padding = 0

        self.group1 = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=64,device=device),
            nn.ReLU(),
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=8, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=64, device=device),
            nn.ReLU(),
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=32, device=device),
            nn.ReLU(),
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=padding),
            nn.BatchNorm2d(num_features=16, device=device),
            nn.ReLU(),
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(64 + 16, 16, kernel_size=1, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=16, device=device),
            nn.ReLU(),
        )

        # Linear layer
        self.linear = nn.Linear(16, 1, device=device)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       x1 = self.group1(x)
       # print(x1.size())
       x2 = self.group2(x)
       # print(x2.size())
       x3 = self.group3(x1)
       x4 = self.group4(x3)
       # print(x3.size())
       x5 = torch.cat((x2, x4), dim=1)
       x6 = self.group5(x5)
       x7 = torch.flatten(x6, start_dim=1)
       x8 = self.linear(x7)
       return self.sigmoid(x8/1000)
    
# DIV 1000!
class CustomNetV3EvenMoreFeatures(nn.Module):
    def __init__(self, device):
        super().__init__()

        stride = 1
        padding = 0

        self.group1 = nn.Sequential(
            nn.Conv2d(13, 128, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=128,device=device),
            nn.ReLU(),
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(13, 128, kernel_size=8, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=128, device=device),
            nn.ReLU(),
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=64, device=device),
            nn.ReLU(),
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=padding),
            nn.BatchNorm2d(num_features=32, device=device),
            nn.ReLU(),
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(128 + 32, 16, kernel_size=1, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=16, device=device),
            nn.ReLU(),
        )

        # Linear layer
        self.linear = nn.Linear(16, 1, device=device)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       x1 = self.group1(x)
       # print(x1.size())
       x2 = self.group2(x)
       # print(x2.size())
       x3 = self.group3(x1)
       x4 = self.group4(x3)
       # print(x3.size())
       x5 = torch.cat((x2, x4), dim=1)
       x6 = self.group5(x5)
       x7 = torch.flatten(x6, start_dim=1)
       x8 = self.linear(x7)
       return self.sigmoid(x8/1000)

# DIV 1000!
class CustomNetV3MoreThanEvenMoreFeatures(nn.Module):
    def __init__(self, device):
        super().__init__()

        stride = 1
        padding = 0

        self.group1 = nn.Sequential(
            nn.Conv2d(13, 256, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=256,device=device),
            nn.ReLU(),
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(13, 256, kernel_size=8, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=256, device=device),
            nn.ReLU(),
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=128, device=device),
            nn.ReLU(),
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=padding),
            nn.BatchNorm2d(num_features=64, device=device),
            nn.ReLU(),
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(256 + 64, 32, kernel_size=1, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=32, device=device),
            nn.ReLU(),
        )

        # Linear layer
        self.linear = nn.Linear(32, 1, device=device)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       x1 = self.group1(x)
       # print(x1.size())
       x2 = self.group2(x)
       # print(x2.size())
       x3 = self.group3(x1)
       x4 = self.group4(x3)
       # print(x3.size())
       x5 = torch.cat((x2, x4), dim=1)
       x6 = self.group5(x5)
       x7 = torch.flatten(x6, start_dim=1)
       x8 = self.linear(x7)
       return self.sigmoid(x8/1000)

class CustomNetV4(nn.Module):
    def __init__(self, device):
        super().__init__()

        stride = 1
        padding = 0

        self.group1 = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=64,device=device),
            nn.ReLU(),
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=128,device=device),
            nn.ReLU(),
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=256,device=device),
            nn.ReLU(),
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=2, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=128,device=device),
            nn.ReLU(),
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=2, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=64,device=device),
            nn.ReLU(),
        )

        self.parallel_group1 = nn.Sequential(
            nn.Conv2d(13, 256, kernel_size=8, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=256, device=device),
            nn.ReLU(),
        )

        self.cat_group1 = nn.Sequential(
            nn.Conv2d(256 + 64, 32, kernel_size=1, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=32, device=device),
            nn.ReLU(),
        )
       
        # Linear layer
        self.linear = nn.Linear(32, 1, device=device)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.group1(x)
        x2 = self.group2(x1)
        x3 = self.group3(x2)
        x4 = self.group4(x3)
        x5 = self.group5(x4)
        x6 = self.parallel_group1(x)
        x7 = torch.cat((x6, x5), dim=1)
        x8 = self.cat_group1(x7)
        x9 = torch.flatten(x8, start_dim=1)
        x10 = self.linear(x9)
        return self.sigmoid(x10/1000)

# DIV 1000!
class CustomNetV3InsaneAmountOfParams(nn.Module):
    def __init__(self, device):
        super().__init__()

        stride = 1
        padding = 0

        self.group1 = nn.Sequential(
            nn.Conv2d(13, 512, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=512,device=device),
            nn.ReLU(),
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(13, 512, kernel_size=8, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=512, device=device),
            nn.ReLU(),
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=256, device=device),
            nn.ReLU(),
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=padding),
            nn.BatchNorm2d(num_features=128, device=device),
            nn.ReLU(),
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(128 + 512, 64, kernel_size=1, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=64, device=device),
            nn.ReLU(),
        )

        # Linear layer
        self.linear = nn.Linear(64, 1, device=device)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       x1 = self.group1(x)
       # print(x1.size())
       x2 = self.group2(x)
       # print(x2.size())
       x3 = self.group3(x1)
       x4 = self.group4(x3)
       # print(x3.size())
       x5 = torch.cat((x2, x4), dim=1)
       x6 = self.group5(x5)
       x7 = torch.flatten(x6, start_dim=1)
       x8 = self.linear(x7)
       return self.sigmoid(x8/1000)
    
# DIV 1000!
class CustomNetV3UltraInsaneAmountOfParams(nn.Module):
    def __init__(self, device):
        super().__init__()

        stride = 1
        padding = 0

        self.group1 = nn.Sequential(
            nn.Conv2d(13, 1024, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=1024,device=device),
            nn.ReLU(),
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(13, 1024, kernel_size=8, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=1024, device=device),
            nn.ReLU(),
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=512, device=device),
            nn.ReLU(),
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=stride, padding=padding, device=device),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=padding),
            nn.BatchNorm2d(num_features=256, device=device),
            nn.ReLU(),
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(256 + 1024, 128, kernel_size=1, stride=stride, padding=padding, device=device),
            nn.BatchNorm2d(num_features=128, device=device),
            nn.ReLU(),
        )

        # Linear layer
        self.linear = nn.Linear(128, 1, device=device)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       x1 = self.group1(x)
       # print(x1.size())
       x2 = self.group2(x)
       # print(x2.size())
       x3 = self.group3(x1)
       x4 = self.group4(x3)
       # print(x3.size())
       x5 = torch.cat((x2, x4), dim=1)
       x6 = self.group5(x5)
       x7 = torch.flatten(x6, start_dim=1)
       x8 = self.linear(x7)
       return self.sigmoid(x8/1000)