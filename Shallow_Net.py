
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Dropout2d(p=0.25),
            nn.Conv2d(4, 40, (62, 30), stride=25, padding=0),
            nn.BatchNorm2d(40),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout2d(p=0.25),
            nn.Conv2d(20, 40, (1,30), stride=25, padding=0),
            nn.BatchNorm2d(40),
            nn.ReLU(True),
        )

        # Layer 2
        self.fc1 = nn.Linear(2*180, 2)

    def forward(self, x):
        out = self.conv1(x)
        #out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


