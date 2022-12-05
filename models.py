import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

STATE_SIZE = 4
ACTION_NUM = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models Definition
# DQN
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(STATE_SIZE, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64*8*8, 512)
        self.fc2 = nn.Linear(512, ACTION_NUM)
        self.device = device

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)

        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            #m.bias.data.fill_(0.1)

    def forward(self, x):
        x = x.to(self.device).float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        q_value = self.fc2(x)
        return q_value

# Dueling DQN
class DuelingDQN(nn.Module):
  def __init__(self):
    super(DuelingDQN, self).__init__()
    self.conv1 = nn.Conv2d(STATE_SIZE, 32, kernel_size=8, stride=4)
    nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
    nn.init.constant_(self.conv1.bias, 0)

    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
    nn.init.constant_(self.conv2.bias, 0)

    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
    nn.init.constant_(self.conv3.bias, 0)

    self.conv4 = nn.Conv2d(64, 1024, kernel_size=7, stride=2)
    nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
    nn.init.constant_(self.conv4.bias, 0)
    # add comment
    self.fc_value = nn.Linear(512, 1)
    nn.init.kaiming_normal_(self.fc_value.weight, nonlinearity='relu')
    self.fc_advantage = nn.Linear(512, ACTION_NUM)
    nn.init.kaiming_normal_(self.fc_advantage.weight, nonlinearity='relu')

  def forward(self, x):
    x = x/255
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    # add comment
    x_value = x[:,:512,:,:].view(-1,512)
    x_advantage = x[:,512:,:,:].view(-1,512)
    x_value = self.fc_value(x_value)
    x_advantage = self.fc_advantage(x_advantage)
    # add comment
    q_value = x_value + x_advantage.sub(torch.mean(x_advantage, 1)[:, None])
    return q_value