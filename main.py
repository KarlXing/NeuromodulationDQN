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

# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython improt display

# turn on pyplot interactive mode
# plt.ion()

# select device (gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


###############################################
# define Transition and ReplayMemory
Transition = namedtuple('Transition', 'state action next_state reward')

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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
# Training
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 20

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(100000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.from_numpy(np.transpose(state,(2,0,1))).float().unsqueeze(0)
            return policy_net(state).max(1)[1][0]
    else:
        return torch.tensor([[random.randrange(9)]], device = device, dtype = torch.long)[0][0]


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                    batch.next_state)), device = device, dtype = torch.uint8)
    non_final_next_states = torch.from_numpy(np.transpose([s for s in batch.next_state if s is not None],
                                    (0,3,1,2))).float()

    state_batch = torch.from_numpy(np.transpose(batch.state,(0,3,1,2))).float()
    reward_batch = torch.tensor(batch.reward)
    action_batch = torch.tensor([[s] for s in batch.action])

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.data.cpu()


num_episodes = 10000000
for i_episode in range(num_episodes):
    env.reset()
    last_obs = env.reset()/255
    overall_reward = 0
    for t in count():
        action = select_action(last_obs)
        obs, reward, done, _ = env.step(action)
        if done:
            obs = None
        else:
            obs = obs/255
        memory.push(last_obs, action, obs, reward)
        last_obs = obs
        loss = optimize_model()
        overall_reward += reward
        if done:
            writer.add_scalar('data/loss', loss, i_episode)
    writer.add_scalar('data/reward', overall_reward, i_episode)
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Complete")
env.render()
env.close()
writer.export_scalars_to_json("./all_scalars.json")
writer.close()





