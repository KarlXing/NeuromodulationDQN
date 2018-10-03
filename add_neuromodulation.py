import numpy as np
import gym
import time
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from tensorboardX import SummaryWriter

# new: define model
class DQN_REP(nn.Module):
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
        x = self.f1(x)
        return x


# delta rule for temporal difference learning
def delta(vnext, v, rwd):
    return rwd + vnext - v;

# Softmax function
#    choose an action based on the actor weights
def action_select (m, beta):

    p = np.zeros(len(m))

    # get softmax probabilities
    for i in range(len(m)):
        p[i] = np.exp(beta*m[i])/sum(np.exp(beta*m))

    r = random.random() # random number for action selection
    sumprob = 0

    # choose an action based on the probability distribution
    i = 0
    done = False
    while not done and i < m.size:
        sumprob = sumprob + p[i];
        if sumprob > r:
            act = i
            done = True
        i += 1

    return act

# Calculate neural activities using a tanh function
#     activity should range from 0 to 1 with a nice "S" curve
def neural_activity(s,mid,g):

    n = np.zeros(len(s))
    for i in range(len(s)):
        n[i] = (np.tanh((s[i]-mid)/g) + 1.0)/2.0

    return n

# new: add state representation function
def state_representation(state, min_range, max_range):
    state = torch.from_numpy(np.transpose(state,(2,0,1))).float().unsqueeze(0).to(device)
    state_rep = dqn_rep(state).squeeze().to('cpu').numpy()
    max_value = np.amax(state_rep)
    min_value = np.amin(state_rep)
    for i in range(state_rep.shape[0]):
        state_rep[i] = (max_range-min_range)/(max_value-min_value)*(state_rep[i]-min_value)+min_range
    return state_rep

# new: add model path
print ("The arguments are: " + str(sys.argv), file=sys.stderr)
if len(sys.argv) != 5:
    print ("ERROR!  Usage is: rlMsPacMan.py learningRate beta mode (0=random; 1=phasic/tonic; 2=low tonic 3=medium tonic; 4 = high tonic) model_path", file=sys.stderr)
    sys.exit()

learningRate = float(sys.argv[1])
beta = float(sys.argv[2])
mode = int(sys.argv[3])
model_path = sys.argv[4]

print ("   Learning rate = " + str(learningRate), file=sys.stderr)
print ("   BETA = " + str(beta), file=sys.stderr)
print ("   Mode = " + str(mode), file=sys.stderr)
print ("   Model Path = " + model_path, file=sys.stderr)


# Using the Atari Ms. Pacman game with RAM
#    RAM is 128 bytes long
#    There a 9 joystick actions associated with this game
# new: new game with pixel states
env = gym.make("MsPacman-v0").unwrapped
reward, info, done = None, None, None
killPenalty = 50.0
# new: update stateSpaceSize and add dqn model
stateSpaceSize = 512
actionSize = 9
with torch.no_grad():
    dqn_rep = DQN_REP()
    dqn_rep.load_state_dict(torch.load(model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

# Actor-Critic model
#    Reward critic predicts values based on rewards from the environment
#    Cost critic predicts the agent being killed
#    Actor takes information from both critics to make a move that gives higher value
#    Weights are normalized to keep in bounds
actions = np.zeros(actionSize)
criticRwd = np.full((stateSpaceSize),1.0)
criticRwd = criticRwd/np.linalg.norm(criticRwd)
criticCost = np.full((stateSpaceSize),1.0)
criticCost = criticCost/np.linalg.norm(criticCost)
actor = np.full((actionSize, stateSpaceSize), 1.0)
for i in range(actionSize):
    actor[i,:] = actor[i,:]/np.linalg.norm(actor[i,:])

n = np.zeros(stateSpaceSize)

# fd = open("Results/rewards.txt", "w")

dCost = 0
dRwd = 0


steps_done = 0
while(True):
    overallReward = 0
    vRwd = 0
    vRwdPre = 0
    vCost = 0
    vCostPre = 0
    action = 0
    state = env.reset()
    state, reward, done, info = env.step(0)
    # use new representation
    state = state_representation(state, 0.0, 255.0)
    livesLeft = info['ale.lives']

    # cnt = 0

    # done is true when the agent loses three lives
    while done != True:
        steps_done += 1
        # actions are chosen based on actor and state of the game
        action = action_select(actions, beta)
        livesLeftPre = livesLeft
        if mode == 0:
            state, reward, done, info = env.step(env.action_space.sample())
            state = state_representation(state, 0.0, 255.0)
        else:
            state, reward, done, info = env.step(action)
            state = state_representation(state, 0.0, 255.0)
    
        vRwdPre = vRwd
        vRwd = 0
        vCostPre = vCost
        vCost = 0.0
        
        # get the state of the RAM, convert it into neural activity
        if mode == 1:
            if np.absolute(dRwd) > 2.0 or np.absolute(dCost) > 2.0:
                n = neural_activity(state,128,100.0)
            else:
                n = neural_activity(state,128,10.0)
        elif mode == 2:
                n = neural_activity(state,128,10.0)
        elif mode == 3:
                n = neural_activity(state,128,50.0)
        else: # mode == 4
                n = neural_activity(state,128,100.0)

        actions = np.zeros(actionSize)

        # calculate the value prediction for reward and cost based on neural state activity
        for i in range(len(state)):
            vRwd = vRwd + criticRwd[i]*n[i]
            vCost = vCost + criticCost[i]*n[i]
            # calculate the actor values based on the neural state
            for j in range(len(actions)):
                actions[j] = actions[j] + actor[j,i]*n[i]
    
        # Use the lives left information to tell whether the agent was killed
        #    If the agent is killed, there is a penalty
        livesLeft = info['ale.lives']
        rwd = reward
        if livesLeftPre == livesLeft:
            cost = 0.0
        else:
            cost = killPenalty
    
        
        # Calculate and apply the delta rule to the reward and cost critic and the actor
        #    Reward critic and actor
        d = delta(vRwd, vRwdPre, rwd)
        dRwd = 0.25*d + 0.75*dRwd
        criticRwd = criticRwd + learningRate * d * n
        criticRwd = criticRwd/np.linalg.norm(criticRwd)
        actor[action,:] = actor[action,:] + learningRate * d * n
        #    Cost critic and actor
        d = -delta(vCost, vCostPre, cost)
        dCost = 0.25*d + 0.75*dCost
        criticCost = criticCost + learningRate * d * n
        criticCost = criticCost/np.linalg.norm(criticCost)
        actor[action,:] = actor[action,:] + learningRate * d * n
        actor[action,:] = actor[action,:]/np.linalg.norm(actor[action,:])

        # keep track of the game score
        overallReward += reward

        # cnt += 1
        # print(str(trial) + "\t" + str(cnt) + "\t" + format(dRwd,"3.5") + "\t" + format(dCost,"3.5") + "\t" + str(vCostPre) + "\n", file=fd)
        # print(str(trial) + "\t" + str(cnt) + "\t" + format(dRwd,"3.5") + "\t" + format(dCost,"3.5") + "\n", file=fd)

        # Uncomment these lines if you want to see the game visualization.  It does slow down the simulation.
        # env.render()
        # time.sleep(0.025)
    write.add_scalor('data/reward', overallReward, steps_done)
    # print the game score to the screen and to a file
    # print (str(overallReward))
    # print (str(trial) + ": " + str(overallReward), file=sys.stderr)

# fd.close()
np.save("/home/jinwei/Documents/Git/NeuromodulationDQN/model_policy/actor",actor)
writer.close()
