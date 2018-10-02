# -*- coding: utf-8 -*-
# This is neuromodulation applied into DQN project

import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from tensorboardX import SummaryWriter


###############################################
# prepare env and device 

env = gym.make('MsPacman-v0').unwrapped
model_path = "/home/jinwei/Documents/Git/NeuromodulationDQN/model/model-0.0001-noskip-basic-0.9-0.02-20000decay-1000steps-600k"
# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython improt display

# turn on pyplot interactive mode
# plt.ion()

# select device (gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps_threshold = 0.02

###############################################
# define model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        self.f1 = nn.Linear(22*16*64, 512)
        self.f2 = nn.Linear(512, 9)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.f1(x))
        x = self.f2(x)
        return x


#############################################
# load model

with torch.no_grad():
    policy_net = DQN().to(device)
    policy_net.load_state_dict(torch.load(model_path))


def select_action(state):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.from_numpy(np.transpose(state,(2,0,1))).float().unsqueeze(0).to(device)
            return policy_net(state).max(1)[1][0].item()
    else:
        return torch.tensor([[random.randrange(9)]])[0][0].item()


num_episodes = 100
rewards = []
for i in range(num_episodes):
    if (i%10 == 0):
        print("finished %d episodes" %(i))
    overall_reward = 0
    done = False
    last_obs = env.reset()
    while(done == False):
        action = select_action(last_obs)
        obs, reward, done, _ = env.step(action)
        overall_reward += reward
        last_obs = obs
    rewards.append(overall_reward)

rewards = [str(s) for s in rewards]
with open("/home/jinwei/Documents/Git/NeuromodulationDQN/evaluation/evaluation_600k.txt","w") as f:
    f.write('\n'.join(rewards))

print("Complete")
env.render()
env.close()