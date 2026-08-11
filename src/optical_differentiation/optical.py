import torch
import torch.nn as nn
import torch.nn.functional as F


class OpticalFrontend(nn.Module):

    def __init__(self):
        super().__init__()

        dx_kernel = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32)
        dy_kernel = dx_kernel.t().clone()

        laplacian_kernel = torch.tensor([[0,  1, 0],
                                          [1, -4, 1],
                                          [0,  1, 0]], dtype=torch.float32)

        blur_kernel = torch.tensor([[1, 2, 1],
                                     [2, 4, 2],
                                     [1, 2, 1]], dtype=torch.float32) / 16.0

        kernels = torch.stack([dx_kernel, dy_kernel, laplacian_kernel, blur_kernel])
        kernels = kernels.unsqueeze(1)  # [4, 1, 3, 3]

        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(kernels)

        self.conv.weight.requires_grad = False

        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        field = self.conv(x)          
        intensity = field ** 2      
        return self.pool(intensity)   

class OpticalModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.frontend = OpticalFrontend()
        self.conv2 = nn.Conv2d(4, 16, kernel_size=5, padding=2)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.frontend(x)                     
        x = self.pool(F.relu(self.conv2(x)))     
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OpticalFrontendDiagonal(nn.Module):

    def __init__(self):
        super().__init__()

        dx_kernel = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32)
        dy_kernel = dx_kernel.t().clone()

        d45_kernel = torch.tensor([[-2, -1, 0],
                                    [-1,  0, 1],
                                    [ 0,  1, 2]], dtype=torch.float32)

        d135_kernel = torch.tensor([[0,  1, 2],
                                     [-1, 0, 1],
                                     [-2, -1, 0]], dtype=torch.float32)

        kernels = torch.stack([dx_kernel, dy_kernel, d45_kernel, d135_kernel])
        kernels = kernels.unsqueeze(1)  # [4, 1, 3, 3]

        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(kernels)
        self.conv.weight.requires_grad = False

        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        field = self.conv(x)
        intensity = field ** 2 
        return self.pool(intensity)


class OpticalModelDiagonal(nn.Module):

    def __init__(self):
        super().__init__()
        self.frontend = OpticalFrontendDiagonal()
        self.conv2 = nn.Conv2d(4, 16, kernel_size=5, padding=2)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.frontend(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OpticalDiagonalOneLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = OpticalFrontendDiagonal()
        self.fc = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x):
        x = self.frontend(x)
        x = x.flatten(1)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-6)
        return self.fc(x)