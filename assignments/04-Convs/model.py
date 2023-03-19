import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    My model class
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        initialization for the layers
        """
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(20 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward
        """
        # print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = self.fc2(x)
        return x
