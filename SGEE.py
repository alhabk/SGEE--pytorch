import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'

parser.add_argument("--env_name", default="BipedalWalker-v3")
parser.add_argument("--ptr", default=0, type=int)
parser.add_argument("--real_lenth", default=0, type=int)
parser.add_argument('--tau',  default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=500, type=int)  # replay buffer size
parser.add_argument('--capacity_g', default=1000000, type=int)  # replay buffer size of G
parser.add_argument('--batch_size', default=100, type=int)  # mini batch size
parser.add_argument('--seed', default=True, type=bool)
parser.add_argument('--random_seed', default=, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=800, type=int) # num of games
parser.add_argument('--num_episode', default=0, type=int)
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('env:', args.env_name)
print('seed:', args.random_seed)
script_name = os.path.basename(__file__)
eps = np.finfo(np.float32).eps
env = NormalizedEnv(gym.make(args.env_name))

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(device) # min value

directory = './exp'+ script_name +'Seed'+str(args.random_seed)+ args.env_name +'./'

def normal_R_V(R_, current_Q, reward):
    R_ = np.array(R_)
    R_ = (R_ - R_.mean()) / (R_.std() + eps.item())
    value = (current_Q - reward).cpu().detach().numpy()
    value = ((value - value.mean()) / (value.std() + eps.item())).mean()
    # R_ is a list and value is a float
    return R_, value

def fill_g(R_, value, storage_g, storage, real_lenth):
    for i in range(0, len(R_))[::-1]:
        if R_[i] >= value:
            storage_g[args.ptr] = storage[-(i+1)]  # R_和storage的顺序是颠倒的，长度相等
            args.ptr = (args.ptr + 1) % real_lenth
    return storage_g

class Replay_buffer():

    def __init__(self, max_size=args.capacity):
        self.storage = []  # Buffer是个多维列表
        self.storage_g = []
        self.max_size = max_size

    def push(self, data):  # 将data顺序存储进buffer
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

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)  # 输出的是一些列action

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)  # net1 actor
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)  # net2 actor
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)  # net1 Q
        self.critic_target = Critic(state_dim, action_dim).to(device)  # net2 Q
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)


        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        # on policy update
        # 计算每个state的discount reward，并添加进data[4]
        dis_r = 0
        x_, y_, u_, r_, d_, R_ = [], [], [], [], [], []
        for i in range(0, len(self.replay_buffer.storage))[::-1]:
            dis_r = dis_r + args.gamma * self.replay_buffer.storage[i][3]
            R_.append(dis_r)
            x, y, u, r, d= self.replay_buffer.storage[i]
            x_.append(x)
            y_.append(y)
            u_.append(u)
            r_.append(r)
            d_.append(d)

        #  在线更新
        x_, y_, u_, r_, d_ = np.array(x_), np.array(y_), np.array(u_), np.array(r_).reshape(-1, 1), \
                             np.array(d_).reshape(-1, 1)
        state = torch.FloatTensor(x_).to(device)
        action = torch.FloatTensor(u_).to(device)
        next_state = torch.FloatTensor(y_).to(device)
        done = torch.FloatTensor(1 - d_).to(device)
        reward = torch.FloatTensor(r_).to(device)

        target_Q = self.critic(next_state, self.actor(next_state))
        target = reward + (done * args.gamma * target_Q).detach()
        current_Q = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()  # 反向传播
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(param.data)

        if len(self.replay_buffer.storage_g) <= args.capacity_g:
            self.replay_buffer.storage_g = self.replay_buffer.storage_g + self.replay_buffer.storage
            self.replay_buffer.storage = []
            args.real_lenth = len(self.replay_buffer.storage_g)
        else:
            R_, value = normal_R_V(R_, current_Q, reward)
            self.replay_buffer.storage_g = fill_g(R_, value, self.replay_buffer.storage_g, self.replay_buffer.storage, args.real_lenth)

        for it in range(args.update_iteration):  # 总循环数200
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # 计算目标Q值
            target_Q = self.critic_target(next_state, self.actor_target(next_state))

            target = reward + (done * args.gamma * target_Q).detach()  # 这里就是TD目标

            # Get current Q estimate
            current_Q = self.critic(state, action)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # actor和critic的更新次数
            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def main():
    agent = DDPG(state_dim, action_dim, max_action)
    ep_r = 0  # 回合奖励
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t >= args.max_length_of_trajectory:  # 学习完一整个序列
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        if args.load:
            agent.load()
        total_step = 0
        for i in range(args.max_episode):
            args.num_episode = i
            if i % args.log_interval == 0:
                agent.save()
            agent.replay_buffer.storage = []
            total_reward = 0
            step = 0
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)  # shape[0]返回行数，也就是actions的个数
                next_state, reward, done, info = env.step(action)
                # env.render()
                agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))

                state = next_state
                if done or len(agent.replay_buffer.storage) == args.capacity:
                    break
                step += 1
                total_reward += reward
            total_step += step+1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
            agent.writer.add_scalar('Reward/steps', total_reward, global_step=total_step)
            agent.update()
    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()
