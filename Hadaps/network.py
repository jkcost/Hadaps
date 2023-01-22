import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class Cnn_network(nn.Module):
    def __init__(self, window_size, data_num, num_classes, lr):
        super(Cnn_network, self).__init__()
        self.inp = window_size
        self.data_num = data_num
        self.output_dim = num_classes
        self.lr = lr
        self.conv1 = nn.Conv1d(1, 5, 2)  # [1402,1,25]->[1402,5,23]
        self.relu1 = nn.ReLU()
        self.max1d1 = nn.MaxPool1d(2, stride=2)  # [1402,5,11]
        self.conv2 = nn.Conv1d(5, 10, 1)  # [1402,5,11] ->[1402,10,9]
        self.relu2 = nn.ReLU()
        self.max1d2 = nn.MaxPool1d(2, stride=2)  # [1402,10,4]

        self.fc1 = nn.Linear(10, 3)

        self.softmax = nn.Softmax(dim=1)
        self.apply(Cnn_network.init_weights)
        # self.network.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = x.reshape(-1, 1, self.inp)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max1d1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max1d2(x)
        x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]))
        x = self.fc1(x)
        x = self.softmax(x)

        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=0.01)
        elif isinstance(m, torch.nn.LSTM):
            for weights in m.all_weights:
                for weight in weights:
                    torch.nn.init_normal_(weight, std=0.01)


class Mlp_network(nn.Module):
    def __init__(self, batch_size, data_num, output_dim, lr, window_size, dropout_rate=0.1):
        super(Mlp_network, self).__init__()

        self.batch_size = batch_size
        # self.num_step = num_step
        self.lr = lr
        self.data_num = data_num
        self.output_dim = output_dim
        self.window_size = window_size

        self._l1_size = 256 * self.window_size
        self._l2_size = 128 * self.window_size
        self._l3_size = 64 * self.window_size
        self._l4_size = 32 * self.window_size
        self._l5_size = 16 * self.window_size
        self.dropout_rate = dropout_rate

        self.network = torch.nn.Sequential(torch.nn.BatchNorm1d(self.window_size),
                                           torch.nn.Linear(self.window_size, self._l1_size),
                                           torch.nn.BatchNorm1d(self._l1_size),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Dropout(p=self.dropout_rate),
                                           torch.nn.Linear(self._l1_size, self._l2_size),
                                           torch.nn.BatchNorm1d(self._l2_size),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Dropout(p=self.dropout_rate),
                                           torch.nn.Linear(self._l2_size, self._l3_size),
                                           torch.nn.BatchNorm1d(self._l3_size),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Dropout(p=self.dropout_rate),
                                           torch.nn.Linear(self._l3_size, self._l4_size),
                                           torch.nn.BatchNorm1d(self._l4_size),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Dropout(p=self.dropout_rate),
                                           torch.nn.Linear(self._l4_size, self._l5_size),
                                           torch.nn.Linear(self._l5_size, self.output_dim),
                                           torch.nn.Softmax(dim=1))  # torch.nn.Softmax(dim=1)
        # self.network.add_module('activation', torch.nn.ReLU())
        self.network.apply(Mlp_network.init_weights)
        # self.network.to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, x):
        # x = x.reshape([self.data_num, -1])
        x = self.network(x)
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=0.01)
        elif isinstance(m, torch.nn.LSTM):
            for weights in m.all_weights:
                for weight in weights:
                    torch.nn.init_normal_(weight, std=0.01)


class ReplayBuffer():
    def __init__(self, batch_size, data_num):
        self.batch_size = batch_size
        self.data_num = data_num
        self.buffer = collections.deque(maxlen=4000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state, action, reward, next_state, done = map(np.stack, zip(*mini_batch))

        # return torch.FloatTensor(state).to(device),torch.FloatTensor(action).to(device),torch.FloatTensor(reward).to(device).unsqueeze(1),torch.FloatTensor(next_state).to(device),torch.FloatTensor(done).to(device).unsqueeze(1)
        return torch.FloatTensor(state), torch.FloatTensor(action), torch.FloatTensor(reward).unsqueeze(
            1), torch.FloatTensor(next_state), torch.FloatTensor(done).unsqueeze(1)



    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, batch_size, data_num, output_dim, lr, window_size, dropout_rate=0.1):
        super(Qnet, self).__init__()

        self.batch_size = batch_size
        # self.num_step = num_step
        self.lr = lr
        self.data_num = data_num
        self.output_dim = output_dim
        self.window_size = window_size

        self._l1_size = 256 * self.window_size
        self._l2_size = 128 * self.window_size
        self._l3_size = 64 * self.window_size
        self._l4_size = 32 * self.window_size
        self._l5_size = 16 * self.window_size
        self.dropout_rate = dropout_rate

        self.network = torch.nn.Sequential(torch.nn.BatchNorm1d(self.data_num * self.window_size),
                                           torch.nn.Linear(self.data_num * self.window_size, self._l1_size),
                                           torch.nn.BatchNorm1d(self._l1_size),
                                           torch.nn.Dropout(p=self.dropout_rate),
                                           torch.nn.Linear(self._l1_size, self._l2_size),
                                           torch.nn.BatchNorm1d(self._l2_size),
                                           torch.nn.Dropout(p=self.dropout_rate),
                                           torch.nn.Linear(self._l2_size, self._l3_size),
                                           torch.nn.BatchNorm1d(self._l3_size),
                                           torch.nn.Dropout(p=self.dropout_rate),
                                           torch.nn.Linear(self._l3_size, self._l4_size),
                                           torch.nn.BatchNorm1d(self._l4_size),
                                           torch.nn.Dropout(p=self.dropout_rate),
                                           torch.nn.Linear(self._l4_size, self._l5_size),
                                           torch.nn.Linear(self._l5_size, (output_dim * (self.data_num + 5))))
        self.network.add_module('activation', torch.nn.LeakyReLU())
        self.network.apply(Qnet.init_weights)
        # self.network.to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, x):
        x = x.reshape([self.batch_size, -1])
        x = self.network(x)
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=0.01)
        elif isinstance(m, torch.nn.LSTM):
            for weights in m.all_weights:
                for weight in weights:
                    torch.nn.init_normal_(weight, std=0.01)

    def sample_action(self, state, epsilon):
        self.network.eval()

        out = self.network(state).reshape(self.data_num+5 , self.output_dim)
        if self.output_dim == 2:
            confidence = out[:, 0]

        else:
            L = out[:, 0]
            S = out[:, 2]
            confidence = L - S


        coin = random.random()
        if coin < epsilon:

            if self.output_dim == 2:
                out = np.random.randint(0, 2, self.data_num +5)
                for i in range(len(out)):
                    if out[i] == 0:
                        out[i] = -1
            else:
                out = np.random.randint(-1, 2, self.data_num +5)
            return np.array(out), confidence.detach().cpu().numpy()
        else:
            if self.output_dim == 2:
                out = out.argmax(axis=1)
                for i in range(len(out)):
                    if out[i] == 0:
                        out[i] = 1
                    else:
                        out[i] = -1
            else:
                out = 1 - out.argmax(axis=1)
            return out.detach().cpu().numpy(), confidence.detach().cpu().numpy()

    def revert_action(self, a_lst):
        revert_action = []
        a_lst = a_lst.tolist()
        if self.output_dim == 3:
            for a in a_lst:
                for i in range(len(a)):
                    if a[i] == -1:
                        a[i] = 0
                    elif a[i] == 0:
                        a[i] = 1
                    elif a[i] == 1:
                        a[i] = 2
                revert_action.append(a)
        else:
            for a in a_lst:
                for i in range(len(a)):
                    if a[i] == -1:
                        a[i] = 0
                    elif a[i] == 1:
                        a[i] = 1
                revert_action.append(a)
        return torch.Tensor(revert_action).type(torch.int64).reshape(self.batch_size, self.data_num+5, 1)


class Policynet(nn.Module):
    def __init__(self, ta_parameter, mode, batch_size, data_num, output_dim, lr, window_size, dropout_rate=0.1,
                 init_alpha=0.001):

        super(Policynet, self).__init__()
        self.ta_parameter = ta_parameter
        self.mode = mode
        self.target_entropy = -1.0  # for automated alpha update
        self.init_alpha = init_alpha
        self.batch_size = batch_size
        # self.num_step = num_step
        self.lr = lr
        self.data_num = data_num
        self.output_dim = output_dim
        self.window_size = window_size

        self._l1_size = 256 * self.window_size
        self._l2_size = 128 * self.window_size
        self._l3_size = 64 * self.window_size
        self._l4_size = 32 * self.window_size
        self._l5_size = 16 * self.window_size
        self.dropout_rate = dropout_rate

        self.network_stock = torch.nn.Sequential(torch.nn.BatchNorm1d(self.data_num * self.window_size),
                                                 torch.nn.Linear(self.data_num * self.window_size, self._l1_size),
                                                 torch.nn.BatchNorm1d(self._l1_size),
                                                 torch.nn.LeakyReLU(),
                                                 torch.nn.Dropout(p=self.dropout_rate),

                                                 torch.nn.Linear(self._l1_size, self._l2_size),
                                                 torch.nn.BatchNorm1d(self._l2_size),
                                                 torch.nn.LeakyReLU(),
                                                 torch.nn.Dropout(p=self.dropout_rate),

                                                 torch.nn.Linear(self._l2_size, self._l3_size),
                                                 torch.nn.BatchNorm1d(self._l3_size),
                                                 torch.nn.LeakyReLU(),
                                                 torch.nn.Dropout(p=self.dropout_rate),

                                                 torch.nn.Linear(self._l3_size, self._l4_size),
                                                 torch.nn.BatchNorm1d(self._l4_size),
                                                 torch.nn.LeakyReLU(),
                                                 torch.nn.Dropout(p=self.dropout_rate),
                                                 torch.nn.Linear(self._l4_size, self._l5_size)
                                                 )

        self.network_crypto = torch.nn.Sequential(torch.nn.BatchNorm1d(self.data_num * self.window_size),
                                                  torch.nn.Linear(self.data_num * self.window_size, self._l1_size),
                                                  torch.nn.BatchNorm1d(self._l1_size),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.Dropout(p=self.dropout_rate),

                                                  torch.nn.Linear(self._l1_size, self._l2_size),
                                                  torch.nn.BatchNorm1d(self._l2_size),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.Dropout(p=self.dropout_rate),

                                                  torch.nn.Linear(self._l2_size, self._l3_size),
                                                  torch.nn.BatchNorm1d(self._l3_size),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.Dropout(p=self.dropout_rate),

                                                  torch.nn.Linear(self._l3_size, self._l4_size),
                                                  torch.nn.BatchNorm1d(self._l4_size),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.Dropout(p=self.dropout_rate),
                                                  torch.nn.Linear(self._l4_size, self._l5_size)
                                                  )

        self.network_portfolio = torch.nn.Sequential(torch.nn.BatchNorm1d(self.data_num * self.window_size),
                                                     torch.nn.Linear(self.data_num * self.window_size, self._l1_size),
                                                     torch.nn.BatchNorm1d(self._l1_size),
                                                     torch.nn.LeakyReLU(),
                                                     torch.nn.Dropout(p=self.dropout_rate),

                                                     torch.nn.Linear(self._l1_size, self._l2_size),
                                                     torch.nn.BatchNorm1d(self._l2_size),
                                                     torch.nn.LeakyReLU(),
                                                     torch.nn.Dropout(p=self.dropout_rate),

                                                     torch.nn.Linear(self._l2_size, self._l3_size),
                                                     torch.nn.BatchNorm1d(self._l3_size),
                                                     torch.nn.LeakyReLU(),
                                                     torch.nn.Dropout(p=self.dropout_rate),

                                                     torch.nn.Linear(self._l3_size, self._l4_size),
                                                     torch.nn.BatchNorm1d(self._l4_size),
                                                     torch.nn.LeakyReLU(),
                                                     torch.nn.Dropout(p=self.dropout_rate),
                                                     torch.nn.Linear(self._l4_size, self._l5_size)
                                                     )

        self.fc_mu_stock = nn.Linear(self._l5_size, 5)
        self.fc_std_stock = nn.Linear(self._l5_size, 5)
        self.fc_mu_crypto = nn.Linear(self._l5_size, 5)
        self.fc_std_crypto = nn.Linear(self._l5_size, 5)
        self.fc_mu_portfolio = nn.Linear(self._l5_size, 3)
        self.fc_std_portfolio = nn.Linear(self._l5_size, 3)

        # self.network_stock.add_module('activation', torch.nn.ReLU())
        self.network_stock.apply(Policynet.init_weights)
        # self.network_crypto.add_module('activation', torch.nn.ReLU())
        self.network_crypto.apply(Policynet.init_weights)
        # self.network_portfolio.add_module('activation', torch.nn.ReLU())
        self.network_portfolio.apply(Policynet.init_weights)
        # self.network.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, s):
        ACTION_SCALE = self.ta_parameter
        x = s[:, 0, :]
        x_variation = s[:, 1, :]
        x_stock = self.network_stock(x)
        stock_mu = self.fc_mu_stock(x_stock)
        stock_std = self.fc_std_stock(x_stock)
        x_crypto = self.network_crypto(x)
        crypto_mu = self.fc_mu_crypto(x_crypto)
        crypto_std = self.fc_std_crypto(x_crypto)
        portfolio_portion = self.network_portfolio(x_variation)
        portfolio_mu = self.fc_mu_portfolio(portfolio_portion)
        portfolio_std = self.fc_std_portfolio(portfolio_portion)

        mu = torch.cat([stock_mu, crypto_mu, portfolio_mu], dim=1)

        log_std = torch.cat([stock_std, crypto_std, portfolio_std], dim=1)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()
        # print(std)
        if self.mode != 'train':
            real_action = torch.tanh(mu)

            real_action = real_action * ACTION_SCALE

            return real_action, 0
        dist = Normal(mu, std)
        action = dist.rsample()

        real_action = torch.tanh(action)
        real_action = real_action * ACTION_SCALE
        log_prob = dist.log_prob(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-6)
        real_log_prob = real_log_prob.sum(1, keepdim=True)

        # return action,log_prob
        return real_action, real_log_prob

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=0.01)
        elif isinstance(m, torch.nn.LSTM):
            for weights in m.all_weights:
                for weight in weights:
                    torch.nn.init_normal_(weight, std=0.01)


class Sac_Qnet(nn.Module):
    def __init__(self, batch_size, data_num, lr, window_size, output_dim, dropout_rate=0.1, init_alpha=0.001):
        super(Sac_Qnet, self).__init__()
        self.tau = 0.01  # for target network soft update
        self.init_alpha = init_alpha
        self.batch_size = batch_size
        # self.num_step = num_step
        self.lr = lr
        self.data_num = data_num
        self.output_dim = output_dim
        self.window_size = window_size

        self._l1_size = 256 * self.window_size
        self._l2_size = 128 * self.window_size
        self._l3_size = 64 * self.window_size
        self._l4_size = 32 * self.window_size
        self._l5_size = 16 * self.window_size
        self.dropout_rate = dropout_rate

        self.fc_s = torch.nn.Sequential(torch.nn.BatchNorm1d(self.data_num * self.window_size),
                                        torch.nn.Linear(self.data_num * self.window_size, self._l1_size),
                                        torch.nn.BatchNorm1d(self._l1_size),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Dropout(p=self.dropout_rate),
                                        # torch.nn.ReLU(),
                                        torch.nn.Linear(self._l1_size, self._l2_size),
                                        torch.nn.BatchNorm1d(self._l2_size),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Dropout(p=self.dropout_rate),
                                        # torch.nn.ReLU(),
                                        torch.nn.Linear(self._l2_size, self._l3_size),
                                        torch.nn.BatchNorm1d(self._l3_size),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Dropout(p=self.dropout_rate),
                                        # torch.nn.ReLU(),
                                        torch.nn.Linear(self._l3_size, self._l4_size),
                                        torch.nn.BatchNorm1d(self._l4_size),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Dropout(p=self.dropout_rate),
                                        torch.nn.Linear(self._l4_size, self._l5_size))

        self.fc_a = torch.nn.Sequential(torch.nn.BatchNorm1d(self.data_num + 3),
                                        # for including cash coumns #self.dat_num
                                        torch.nn.Linear(self.data_num + 3, self._l1_size),
                                        torch.nn.BatchNorm1d(self._l1_size),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Dropout(p=self.dropout_rate),
                                        # torch.nn.ReLU(),
                                        torch.nn.Linear(self._l1_size, self._l2_size),
                                        torch.nn.BatchNorm1d(self._l2_size),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Dropout(p=self.dropout_rate),
                                        # torch.nn.ReLU(),
                                        torch.nn.Linear(self._l2_size, self._l3_size),
                                        torch.nn.BatchNorm1d(self._l3_size),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Dropout(p=self.dropout_rate),
                                        # torch.nn.ReLU(),
                                        torch.nn.Linear(self._l3_size, self._l4_size),
                                        torch.nn.BatchNorm1d(self._l4_size),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Dropout(p=self.dropout_rate),
                                        torch.nn.Linear(self._l4_size, self._l5_size)
                                        )

        self.fc_cat = nn.Linear(self._l5_size * 2, 16)
        self.fc_out = nn.Linear(16, 1)

        # self.fc_s.add_module('activation', torch.nn.ReLU())
        self.fc_s.apply(Sac_Qnet.init_weights)
        # self.fc_s.to(device)

        # self.fc_a.add_module('activation', torch.nn.ReLU())
        self.fc_a.apply(Sac_Qnet.init_weights)
        # self.fc_a.to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, a):
        x = x[:, 0, :]
        h1 = self.fc_s(x)
        h2 = self.fc_a(a)
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=0.01)
        elif isinstance(m, torch.nn.LSTM):
            for weights in m.all_weights:
                for weight in weights:
                    torch.nn.init_normal_(weight, std=0.01)
