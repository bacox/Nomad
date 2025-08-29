import torch.nn.functional as F
from torch import Tensor, nn
class CNNMNIST(nn.Module):
    def __init__(self, input_channels: int = 1, output_channels: int = 10, kernel_size: int = 5) -> None:
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, output_channels)
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
