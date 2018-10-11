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

##############################################
# Config

bias_exist = False
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.02
EPS_DECAY = 20000
STEPS_UPDATE = 20
LR = 0.0001
model_path = "/home/jinwei/Documents/Git/NeuromodulationDQN/model/model_rnn"
dqn_path = "/home/jinwei/Documents/Git/NeuromodulationDQN/saved_models/rep.model"
rnnf2_path = "/home/jinwei/Documents/Git/NeuromodulationDQN/saved_models/f2.model"
ACTION_SPACE = 9


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
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.f1(x))
        return x

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        # weights
        self.f2 = nn.Linear(512, ACTION_SPACE, bias=False)
        self.e2e = nn.Linear(ACTION_SPACE, ACTION_SPACE, bias = False)
        self.e2i = nn.Linear(ACTION_SPACE, ACTION_SPACE, bias = False)

    def forward(self, x, E, E_p, mask):
        # clamp weights
        self.e2e = torch.clamp(self.e2e, min = 0)
        self.e2i = torch.clamp(self.e2i, max = 0)
        self.e2e.weights = self.e2e.weights*mask
        self.e2i.weights = self.e2i.weights*mask

        # process
        x = self.f2(x)
        E_n = F.relu(torch.mm(E, self.e2e * self.mask) + x + torch.mm(E_p, self.e2i * self.mask))
        return E_n, E


#############################################
# Training
with torch.no_grad():
    dqn_net = DQN().to(device)
dqn_net.load_state_dict(torch.load(dqn_path))

policy_net = RNN().to(device)
policy_net.f2.load_state_dict(torch.load(rnnf2_path))
with torch.no_grad():
    target_net = RNN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

net2device(policy_net, device)
net2device(target_net, device)

optimizer = optim.RMSprop(policy_net.parameters(), lr = LR)
memory = ReplayMemory(100000)

steps_done = 0


# def select_action(state):
#     global steps_done
#     sample = random.random()
#     eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
#     steps_done += 1
#     if sample > eps_threshold:
#         with torch.no_grad():
#             state = torch.from_numpy(np.transpose(state,(2,0,1))).float().unsqueeze(0).to(device)
#             return [policy_net(dqn(state)).max(1)[1][0].item(), eps_threshold]
#     else:
#         return [torch.tensor([[random.randrange(9)]])[0][0].item(), eps_threshold]

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    with torch.no_grad():
        state = torch.from_numpy(np.transpose(state,(2,0,1))).float().unsqueeze(0).to(device)
        values = policy_net(dqn_net(state))
        print(values)
        print(values.shape)
    if sample > eps_threshold:
        action = values.max(1)[1][0].item()
    else:
        action = torch.tensor([[random.randrange(9)]])[0][0].item()
    return action, eps_threshold, values.squeeze()[action].item()


def delta(vRWd, vRwdPre, reward):
    return vRwd - vRwdPre + reward

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                    batch.next_state)), device = device, dtype = torch.uint8)
    non_final_next_states = torch.from_numpy(np.transpose(np.asarray([s for s in batch.next_state if s is not None]),
                                    (0,3,1,2))).float().to(device)
    state_batch = torch.from_numpy(np.transpose(batch.state,(0,3,1,2))).float().to(device)
    reward_batch = torch.tensor(batch.reward, device = device)
    action_batch = torch.tensor([[s] for s in batch.action], device = device)
    state_action_values = policy_net(dqn_net(state_batch)).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(dqn_net(non_final_next_states)).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

vRwd = None
vRwdPre = None
dRwd = 0

num_episodes = 10000000
i_episode = 0
overall_reward = 0
last_obs = env.reset()
for t in count():
    if (i_episode > num_episodes):
        break
    print(last_obs.shape)
    action, eps_threshold, vRwd = select_action(last_obs)
    obs, reward, done, _ = env.step(action)

    if vRwdPre is not None:
        dRwd = 0.75*dRwd + 0.25*delta(vRwd, vRwdPre, reward)
    vRwdPre = vRwd

    if done:
        obs = None
    memory.push(last_obs, action, obs, reward)
    last_obs = obs
    overall_reward += reward
    optimize_model()
    # if loss_cpu is not None:
    #     writer.add_scalar('data/loss', loss_cpu, steps_done)
    writer.add_scalar('parameters/dRwd', abs(dRwd), steps_done)
    if (steps_done % 1000 == 0):
        with torch.no_grad():
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(policy_net.state_dict(), model_path)
    if done:
        writer.add_scalar('data/reward', overall_reward, steps_done)
        writer.add_scalar('data/eps_threshold', eps_threshold, steps_done)
        last_obs = env.reset()
        i_episode += 1
        overall_reward = 0
        # if (i_episode % 10 == 0):
        #     target_net.load_state_dict(policy_net.state_dict())
        #     torch.save(policy_net.state_dict(), "/home/jinwei/Documents/Git/NeuromodulationDQN/model/model")
        if (i_episode % 100 == 0):
            print(str(i_episode)+" episodes")


print("Complete")
env.render()
env.close()
writer.close()

