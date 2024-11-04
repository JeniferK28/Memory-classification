class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.preLayers = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5), stride=1, padding=0),
            nn.BatchNorm2d(25),

        )
        self.spatial = nn.Sequential(
            nn.Conv2d(25, 25, (124, 1), stride=1, padding=0),
            nn.BatchNorm2d(25),
            nn.ELU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=0)
        self.conv1 = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(25, 50, (1, 5), stride=1, padding=0),
            nn.BatchNorm2d(50),
            nn.ELU(True),
        )
        self.conv2= nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(50, 100, (1, 10), stride=1, padding=0),
            nn.BatchNorm2d(100),
            nn.ELU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(100, 200, (1, 10), stride=1, padding=0),
            nn.BatchNorm2d(200),
            nn.ELU(True),
        )
        self.fc1 = nn.Linear(300*1*4,6)



    def forward(self, x):
        out = self.preLayers(x)
        out = self.spatial(out)
        #out = self.maxpool(out)
        out = self.conv1(out)
        #out = self.maxpool(out)
        out = self.conv2(out)
        #out = self.maxpool(out)
        out = self.conv3(out)
        #out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
