import argparse
from collections import namedtuple
from itertools import count

import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from normalized_env import NormalizedEnv
from torch.distributions import Normal
from tensorboardX import SummaryWriter

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument("--env_name", default="BipedalWalker-v3")  # OpenAI gym environment name， BipedalWalker-v2
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--iteration', default=5, type=int)

parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=5000, type=int)
parser.add_argument('--capacity_g', default=1000000, type=int)
parser.add_argument('--real_lenth', default=0, type=int)
parser.add_argument('--ptr', default=0, type=int)
parser.add_argument('--num_iteration', default=10000, type=int) #  num of  games
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=100, type=int)

# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.2, type=float)
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=2000, type=int)
parser.add_argument('--print_log', default=5, type=int)
args = parser.parse_args()



# Set seeds
env = NormalizedEnv(gym.make(args.env_name))
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
script_name = os.path.basename(__file__)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(device) # min value
directory = './exp' + script_name + args.env_name +'./'
eps = np.finfo(np.float32).eps

def normal_R_V(R_, current_Q, reward):
    R_ = np.array(R_)
    R_ = (R_ - R_.mean()) / (R_.std() + eps.item())
    value = (current_Q - reward).cpu().detach().numpy()
    value = ((value - value.mean()) / (value.std() + eps.item())).mean()
    # R_ is a list and value is a float
    return R_, value

def fill_g(R_, value, storage_g, storage, real_lenth):
    if np.random.uniform()<0.6:
        for i in range(0, len(R_))[::-1]:
            if R_[i] >= value:
                storage_g[args.ptr] = storage[-(i+1)]  # R_和storage的顺序是颠倒的，长度相等
                args.ptr = (args.ptr + 1) % real_lenth
    else:
        for i in range(0, len(R_))[::-1]:
            storage_g[args.ptr] = storage[-(i + 1)]
            args.ptr = (args.ptr + 1) % real_lenth

    return storage_g


class Replay_buffer():

    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.storage_g = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage_g), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage_g[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TD3():

    def __init__(self, state_dim, action_dim, max_action):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters())

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(args.capacity)
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float().to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, num_iteration):
        dis_r = 0
        x_, y_, u_, r_, d_, R_ = [], [], [], [], [], []
        for i in range(0, len(self.memory.storage))[::-1]:
            dis_r = dis_r + args.gamma * self.memory.storage[i][3]
            R_.append(dis_r)
            x, y, u, r, d = self.memory.storage[i]
            x_.append(x)
            # y_.append(y)
            u_.append(u)
            r_.append(r)
            # d_.append(d)

        #  在线更新
        # x_, y_, u_, r_, d_ = np.array(x_), np.array(y_), np.array(u_), np.array(r_).reshape(-1, 1), np.array(d_).reshape(-1, 1)
        x_, u_, r_ = np.array(x_), np.array(u_), np.array(r_).reshape(-1, 1)
        state = torch.FloatTensor(x_).to(device)
        action = torch.FloatTensor(u_).to(device)
        # next_state = torch.FloatTensor(y_).to(device)
        # done = torch.FloatTensor(1 - d_).to(device)
        reward = torch.FloatTensor(r_).to(device)

        current_Q1 = self.critic_1(state, action)
        R_, value = normal_R_V(R_, current_Q1, reward)
        self.memory.storage_g = fill_g(R_, value, self.memory.storage_g, self.memory.storage, args.real_lenth)
        self.memory.storage = []

        for i in range(num_iteration):
            x, y, u, r, d = self.memory.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(device)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            # Delayed policy updates:
            if i % args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory+'actor.pth')
        torch.save(self.actor_target.state_dict(), directory+'actor_target.pth')
        torch.save(self.critic_1.state_dict(), directory+'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), directory+'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), directory+'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), directory+'critic_2_target.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    agent = TD3(state_dim, action_dim, max_action)
    ep_r = 0

    if args.mode == 'test':
        agent.load()
        for i in range(args.iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t ==2000 :
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    break
                state = next_state

    elif args.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        while len(agent.memory.storage_g) <= args.capacity_g:
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                action = action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])
                action = action.clip(env.action_space.low, env.action_space.high)
                next_state, reward, done, info = env.step(action)
                agent.memory.storage_g.append((state, next_state, action, reward, np.float(done)))
                state = next_state
                if done:
                    break
        args.real_lenth = len(agent.memory.storage_g)
        print('##############################')
        print('            开始训练            ')
        print('##############################')

        if args.load:
            agent.load()
        total_step = 0
        for i in range(args.num_iteration):
            step = 0
            state = env.reset()
            ep_r = 0
            for t in count():
                action = agent.select_action(state)
                action = action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])
                action = action.clip(env.action_space.low, env.action_space.high)
                next_state, reward, done, info = env.step(action)
                ep_r += reward
                # if args.render and i >= args.render_interval : env.render()
                agent.memory.push((state, next_state, action, reward, np.float(done)))
                state = next_state
                step += 1
                if done:
                    break
            agent.update(200)
            total_step += step
            agent.writer.add_scalar('Reward', ep_r, global_step=total_step)
            print("Ep_i \t{}, the ep_r is \t{:0.2f}, the total_step is \t{}".format(i, ep_r, total_step))

            if i % args.log_interval == 0:
                agent.save()

    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()
