import random
from collections import namedtuple
from PIL import Image
import argparse
from itertools import count
import tqdm
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from models import *

# Runtime identifier based on the current time
import time
import datetime
def get_time():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')

RUNTIME_IDENTIFIER = get_time()

# Make directory for saving outputs with RUNTIME_IDENTIFIER
if not os.path.exists(RUNTIME_IDENTIFIER):
    os.makedirs(RUNTIME_IDENTIFIER)

# Arguments setup
# --model: model to use (default: DQN)
parser = argparse.ArgumentParser(description='PyTorch Breakout')
parser.add_argument('--model', type=str, default='dqn', help='model to use (default: DuelingDQN)')
parser.add_argument('--size', type=int, default=96, help='size of the frame (default: 84)')
parser.add_argument('--goal', type=str, default='episode', help='goal of the training (default: episode) (episode, reward')
parser.add_argument('--goalvalue', type=int, default=10000, help='goal value of the training (default: 1000)')
args = parser.parse_args()

# Hyperparameters
STATE_SIZE = 4
STATE_W = args.size
STATE_H = args.size
MEMSIZE = 50000
LR = 1e-4
NUM_EPISODES = 100000
OPTIMIZE_MODEL_STEP = 4
TARGET_UPDATE=10000
STEPS_BEFORE_TRAIN = 50000
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000000
eps_threshold = EPS_START
steps_done = 0
BATCH_SIZE = 32
GAMMA = 0.99
mean_size = 100
mean_step = 1

print(f'model used : {args.model}')

# Environment setup
env = gym.make('BreakoutDeterministic-v4').unwrapped

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : {}".format(device))

ACTION_NUM = env.action_space.n
print("Number of actions : {}".format(ACTION_NUM))

class StateHolder:
  def __init__(self, number_screens=4):
    self.first_action=True
    self.state = torch.ByteTensor(1, args.size, args.size).to(device)
    self.number_screens = number_screens

  def push(self, screen):
    new_screen = screen.squeeze(0)
    if self.first_action:
      self.state[0] = new_screen
      for number in range(self.number_screens-1):
        self.state = torch.cat((self.state, new_screen), 0)
      self.first_action = False
    else:
      self.state = torch.cat((self.state, new_screen), 0)[1:]

  def get(self):
    return self.state.unsqueeze(0)

  def reset(self):
    self.first_action = True
    self.state = torch.ByteTensor(1, args.size, args.size).to(device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
  def __init__(self, capacity=MEMSIZE):
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


# Get screen -> Function to get state from environment
resize = T.Compose([T.ToPILImage(),
                    T.Resize((STATE_W, STATE_H), interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
  screen = env.render(mode='rgb_array')
  screen = np.dot(screen[...,:3], [0.299, 0.587, 0.114])
  screen = screen[30:195,:]
  screen = np.ascontiguousarray(screen, dtype=np.uint8).reshape(screen.shape[0], screen.shape[1], 1)
  return resize(screen).mul(255).type(torch.ByteTensor).to(device).detach().unsqueeze(0)

# Model Selection
if args.model == 'dqn':
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
elif args.model == 'duelingdqn':
    policy_net = DuelingDQN().to(device)
    target_net = DuelingDQN().to(device)
elif args.model == 'coatdqn':
    print('Comming soon!') #TBA in models.py
else:
    print('Model not found')
    exit()

target_net.load_state_dict(policy_net.state_dict())
policy_net.train()
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=LR)

# Select Action
def select_action(state, eps_threshold):
    global steps_done
    sample = random.random()
    if sample > eps_threshold and state is not None:
        with torch.no_grad():
            # ret= policy_net(state.float()).max(1)[1].view(1, 1)
            # print(ret)
            return policy_net(state.float()).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(ACTION_NUM)]], device=device, dtype=torch.long)

def select_action_eval(state):
    with torch.no_grad():
        return policy_net(state.float()).max(1)[1].view(1, 1)

# Plot Reward
train_rewards = [] # list of total rewards per episode
test_rewards = [] # list of total rewards per 1000 episode
def plot_rewards(rewards = train_rewards):
    plt.figure(2)
    plt.clf()
    plt.title('Train and Test Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(rewards)
    if len(rewards) > mean_size:
        means = np.array([rewards[i:i+mean_size:] for i in range(0, len(rewards) - mean_size, mean_step)]).mean(1)
        means = np.concatenate((np.zeros(mean_size - 1), means))
        plt.plot(means)
    # Save as PNG
    plt.savefig(f"{RUNTIME_IDENTIFIER}/reward_history_{RUNTIME_IDENTIFIER}.png")

# Initialization
state_holder = StateHolder()
memory = ReplayMemory()
steps_done = 0

# Optimize model
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).float().to(device)
    state_batch = torch.cat(batch.state).float().to(device)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_action = policy_net(non_final_next_states).detach().max(1)[1].view(-1,1)
    next_state_values[non_final_mask] = target_net(non_final_next_states).detach().gather(1, next_state_action).view(-1)

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    del non_final_mask
    del next_state_action
    del non_final_next_states
    del state_batch
    del action_batch
    del reward_batch
    del state_action_values
    del next_state_values
    del expected_state_action_values
    del loss

def print_n_logging(log):
    print(log)
    f = open(f"{RUNTIME_IDENTIFIER}/train_log.txt",'a')
    f.write(log)
    f.write('\n')
    f.close()

print("Start training")

since = time.time()

for e in tqdm.tqdm(range(NUM_EPISODES)):
    env.reset()
    lives = 5
    ep_rewards = []
    state_holder.push(get_screen())

    for t in count():
        state = state_holder.get()

        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        action = select_action(state, eps_threshold)
        steps_done += 1


        _, reward, done, info = env.step(action.item())
        # life = info['ale.lives']
        ep_rewards.append(reward)
        reward = torch.tensor([reward], device=device)

        state_holder.push(get_screen())
        next_state = state_holder.get()

        if not done:
            new_reward = reward
            # next_state, lives = (None, life) if life < lives else (next_state, lives)
            memory.push(state.to('cpu'), action, next_state, new_reward)
            state = next_state
        else:
            next_state = None
            new_reward = torch.zeros_like(reward)
            memory.push(state.to('cpu'), action, next_state, new_reward)
            state = next_state


        if (steps_done > STEPS_BEFORE_TRAIN) and steps_done % OPTIMIZE_MODEL_STEP == 0:
            BATCH_SIZE = 32
            optimize_model()
        if e % 100 == 99 and  t == 0:
            # print('\neps_threshold:', eps_threshold)
            print_n_logging(f'\neps_threshold: {eps_threshold}')
            print_n_logging(f'steps_done: {steps_done}')
            # print('steps_done: ', steps_done)

            # print("Mean score : {}".format(np.mean(train_rewards[-100:])))
            print_n_logging(f'Mean score: {np.mean(train_rewards[-100:])}')
        if e % 10 == 9 and  t == 0:
            print_n_logging(f"10 ep.mean score : {np.mean(train_rewards[-10:])}")
            # print("10 ep.mean score : {}".format(np.mean(train_rewards[-10:])))
        if t > 18000:
            break

        if steps_done % TARGET_UPDATE == 0:
            print_n_logging("Target net updated!")
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            train_rewards.append(np.sum(ep_rewards))
            break

        # Check if goal is reached
        if args.goal == 'episode':
            if e == args.goal_value:
                break
        elif args.goal == 'reward':
            if np.mean(train_rewards[-100:]) > args.goal_value:
                break

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
plot_rewards(train_rewards)
torch.save(policy_net.state_dict(), f"{RUNTIME_IDENTIFIER}/{RUNTIME_IDENTIFIER}_weight.pt")