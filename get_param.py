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

import modulation_utils

###############################################
# prepare env and device 
env = gym.make('MsPacman-v0').unwrapped

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
    def forward(self, x, beta):
        x = modulation_utils.tanh_beta(self.conv1(x), 1)
        x = modulation_utils.tanh_beta(self.conv2(x), 1)
        x = modulation_utils.tanh_beta(self.conv3(x), 1)
        x = x.view(x.size(0), -1)
        x = modulation_utils.tanh_beta(self.f1(x), 1)
        x = self.f2(x)
        return x


#############################################
# Training Setup
BATCH_SIZE = 32
GAMMA = 0.999
STEPS_UPDATE = 1000
SCALE = 5
ACTION_SIZE = 9

policy_net = DQN().to(device)
with torch.no_grad():
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.RMSprop(policy_net.parameters(), lr = 0.0001)
memory = ReplayMemory(100000)
steps_done = 0

#############################################
# Define functions
def select_action(state):
    global steps_done
    steps_done += 1
    with torch.no_grad():
        state = torch.from_numpy(np.transpose(state,(2,0,1))).float().unsqueeze(0).to(device)
        values = policy_net(state).squeeze()
        # scaled_values = SCALE/torch.max(values)*values
        scaled_values = values
        prob = torch.softmax(scaled_values, 0).detach().numpy()
        action = np.random.choice(ACTION_SIZE, 1, p = prob)[0]
        return action, values[action].item(), torch.max(values).detach().item(), torch.mean(values).detach().item()

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
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



####################################
# Training

num_episodes = 10000000
i_episode = 0
overall_reward = 0

# get initial lives
env.reset()
_, _, _, info = env.step(0)
initialLivesLeft = info['ale.lives']

livesLeft = initialLivesLeft
last_obs = env.reset()
vRwd = None
vRwdPre = None
dRwd = 0
for t in count():
    if (i_episode > num_episodes):
        break
    # select action and calc vRwd, dRwd
    action, vRwd, max_out, mean_out = select_action(last_obs)
    obs, reward, done, info = env.step(action)
    if vRwdPre is not None:
        dRwd = 0.75*dRwd + delta(vRwd, vRwdPre, reward)
    vRwdPre = vRwd

    # record output stats
    writer.add_scalar('parameters/max_out', max_out, steps_done)
    writer.add_scalar('parameters/mean_out', mean_out, steps_done) 
    # record dRwd
    writer.add_scalar('parameters/dRwd', dRwd, steps_done)
    if livesLeft != info['ale.lives']:
        writer.add_scalar('parameters/dRwd_dead', dRwd, steps_done)
        livesLeft = info['ale.lives']
    if reward > 0:
        writer.add_scalar('parameters/dRwd_reward', dRwd, steps_done)

    # update memory and optimize model
    if done:
        obs = None
    memory.push(last_obs, action, obs, reward)
    last_obs = obs
    overall_reward += reward
    optimize_model()

    # synchronize
    if (steps_done % STEPS_UPDATE == 0):
        with torch.no_grad():
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(policy_net.state_dict(), "/home/jinwei/Documents/Git/NeuromodulationDQN/model/model")
            writer.export_scalars_to_json("./all_scalars.json")

    # restart and record overall reward on tensorboardX
    if done:
        writer.add_scalar('data/reward', overall_reward, steps_done)
        last_obs = env.reset()
        i_episode += 1
        overall_reward = 0
        livesLeft = initialLivesLeft
        # if (i_episode % 10 == 0):
        #     target_net.load_state_dict(policy_net.state_dict())
        #     torch.save(policy_net.state_dict(), "/home/jinwei/Documents/Git/NeuromodulationDQN/model/model")
        if (i_episode % 100 == 0):
            print(str(i_episode)+" episodes")
    

print("Complete")
env.render()
env.close()
writer.export_scalars_to_json("./all_scalars.json")
writer.close()





