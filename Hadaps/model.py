import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import numpy as np
from torch.distributions import Normal
# import quantstats as qs
from network import  Qnet,ReplayBuffer,Policynet,Sac_Qnet
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
from visualizer import Visualizer
from metrics import Metrics
#Tesnorboard
writer = SummaryWriter('runs/{}_SAC'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
LOGGER_NAME = 'Portfolio'
logger = logging.getLogger(LOGGER_NAME)



class Heuristic():
    def __init__(self,env,args):
        # for cuda
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_num = env.data_num
        self.env = env
        self.args = args
        self.index = 0
        self.monkey_invest_dict = {}
        self.fair_invest_dict = {}

    def monkey_invest(self):
        if self.args.mode =='train':
            data = self.env.train__return_data

        elif self.args.mode =='valid':
            data = self.env.valid_return_data

        else:
            data = self.env.test_return_data

        self.index =0
        done = False

        while not done:
            self.index +=1
            x =np.random.rand(self.data_num+1)
            f_x = np.exp(x)/np.sum(np.exp(x))
            portfolio_value = (f_x* np.array(data.iloc[self.index + self.args.window_size - 1])).sum()
            portfolio_info = {f'{data.index[self.index + self.args.window_size - 1]}': [portfolio_value] + [i for i in f_x]}
            self.monkey_invest_dict.update(portfolio_info)
            if self.index == (len(data) - self.args.window_size):
                done = True


            if done:
                columns = ['Portfolio_value'] + self.env.train_data_lst
                result_data = pd.DataFrame.from_dict(self.monkey_invest_dict, orient='index')
                result_data.columns = columns
                result_data.index = pd.to_datetime(result_data.index)
                returns = result_data['Portfolio_value']
                cm_return = (returns + 1).cumprod() - 1
                result_data['cm_return'] = cm_return

        return result_data['Portfolio_value']
    def fair_invest(self):

        if self.args.mode == 'train':
            data = self.env.train__return_data

        elif self.args.mode == 'valid':
            data = self.env.valid_return_data

        else:
            data = self.env.test_return_data

        done = False
        self.index = 0
        while not done:
            self.index += 1

            f_x = np.array([1/(self.data_num +1 )]*(self.data_num +1))

            portfolio_value = (f_x * np.array(data.iloc[self.index + self.args.window_size - 1])).sum()
            portfolio_info = {
                f'{data.index[self.index + self.args.window_size - 1]}': [portfolio_value] + [i for i in f_x]}
            self.fair_invest_dict.update(portfolio_info)
            if self.index == (len(data) - self.args.window_size):
                done = True

            if done:
                columns = ['Portfolio_value'] + self.env.train_data_lst
                result_data = pd.DataFrame.from_dict(self.fair_invest_dict, orient='index')
                result_data.columns = columns
                result_data.index = pd.to_datetime(result_data.index)
                returns = result_data['Portfolio_value']
                cm_return = (returns + 1).cumprod() - 1
                result_data['cm_return'] = cm_return
        return result_data['Portfolio_value']

class SAC():
    def __init__(self,data_num,output_path,model_path,args):
        # for cuda
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.output_path = output_path
        self.data_num = data_num
        self.batch_size = args.batch_size

        self.gamma = args.discount_factor
        self.tau = args.tau
        self.alpha = args.alpha

        self.lr = args.lr
        self.window_size = args.window_size
        self.output_dim = args.output_dim
        self.mode = args.mode
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.updates_per_step = args.updates_per_step


        if self.automatic_entropy_tuning is True:
            # for automated alpha update, # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            #  self.target_entropy = -torch.prod(torch.Tensor((data_num,)).to(self.device)).item()
             self.target_entropy = -torch.prod(torch.Tensor((data_num,))).item()
            #  self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
             self.log_alpha = torch.zeros(1, requires_grad=True)
             self.alpha_optim = optim.Adam([self.log_alpha], lr=args.lr)

    def calc_target(self,pi, q1, q2, mini_batch):
        s, a, r, s_prime, done = mini_batch


        with torch.no_grad():
            # s_prime = s_prime.reshape([self.batch_size,-1])
            ######################################
            # forward_x = s_prime.split(self.data_num, dim=2)
            # x = forward_x[0].reshape([self.batch_size, -1])
            # x_variation = forward_x[1].reshape([self.batch_size, -1])
            ################################
            a_prime, log_prob = pi(s_prime)

            entropy = (-self.alpha * log_prob)
            q1_val, q2_val = q1(s_prime, a_prime), q2(s_prime, a_prime)

            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            target = r + self.gamma * done * (min_q + entropy)

        return target

    def train_Qnet(self,network,target,mini_batch):
        s,a,r,s_prime,done = mini_batch
        ##########################
        loss = F.mse_loss(network.forward(s,a),target)
        network.optimizer.zero_grad()
        loss.backward()
        network.optimizer.step()
        return loss

    def soft_update(self,network,target_network):
        for param_target,param in zip(target_network.parameters(),network.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def train_pinet(self,network,q1,q2,mini_batch):
        s,_,_,_,_ = mini_batch



        a, log_prob = network.forward(s)
        entropy = (-self.alpha * log_prob)
        q1_val,q2_val = q1(s,a),q2(s,a)
        q1_q2 = torch.cat([q1_val,q2_val], dim=1)
        min_q = torch.min(q1_q2,1,keepdim=True)[0]
        loss = (-min_q - entropy).mean() #for gradient ascent
        network.optimizer.zero_grad()
        loss.backward()
        network.optimizer.step()
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        return loss,alpha_loss.item(),alpha_tlogs.item()

    def train(self,env):
        memory = ReplayBuffer(self.batch_size,self.data_num)
        q1, q2, q1_target, q2_target = Sac_Qnet(self.batch_size, env.data_num, lr=self.lr,
                                                window_size=self.window_size, output_dim=self.output_dim),Sac_Qnet(self.batch_size, env.data_num, lr=self.lr,
                                                window_size=self.window_size, output_dim=self.output_dim),Sac_Qnet(self.batch_size, env.data_num, lr=self.lr,
                                                window_size=self.window_size, output_dim=self.output_dim),Sac_Qnet(self.batch_size, env.data_num, lr=self.lr,
                                                window_size=self.window_size, output_dim=self.output_dim)
        pi = Policynet(self.mode,self.batch_size, env.data_num,  lr=self.lr,
                       window_size=self.window_size, output_dim=self.output_dim)
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())
        print_interval = 30
        episode_reward_lst = []


        updates = 0
        for n_epi in range(301):
            episode_reward = 0.0

            result_dict = {}
            result_data = pd.DataFrame()

            s = env.reset(env.train_data)

            done = False

            while not done:

                pf_data_index = 0
                pi.eval()
                a, _, = pi.forward(s)
                a = a.detach().numpy().squeeze()


                portfolio_portion = np.exp(a[-3:]) / np.sum(np.exp(a[-3:]))
                stock_a = np.exp(a[:5]) / np.sum(np.exp(a[:5]))
                crypto_a = np.exp(a[5:10]) / np.sum(np.exp(a[5:10]))

                stock_a = stock_a * portfolio_portion[1]
                crypto_a = crypto_a * portfolio_portion[2]
                confidence = np.concatenate((stock_a, crypto_a))
                confidence = np.append(confidence,portfolio_portion[0])

                pf_data_index += 1

                s_prime, r, done, portfolio_dict = env.step(env.train_data, a, confidence, self.mode)
                result_dict.update(portfolio_dict)

                done_mask = 0.0 if done else 1.0

                memory.put((s.squeeze(), a, r, s_prime.squeeze(), done_mask))

                s = s_prime
                episode_reward += r
                if done:
                    episode_reward_lst.append(episode_reward)
                    writer.add_scalar('reward per episode', episode_reward, n_epi)
                    columns = ['Portfolio_value'] + env.train_data_lst
                    result_data = pd.DataFrame.from_dict(result_dict, orient='index')
                    result_data.columns = columns
                    if n_epi < 200:
                        logger.info(f'epoch:{n_epi}->reward({episode_reward})')
                        logger.info(f'epoch:{n_epi}-> alpha({self.alpha})')
                    break

            if memory.size() > 2000:
                check_policy_loss = 0.0
                check_q1_loss = 0.0
                check_q2_loss = 0.0

                for i in range(self.updates_per_step):
                    mini_batch = memory.sample(self.batch_size)
                    td_target = self.calc_target(pi,q1_target,q2_target,mini_batch)
                    q1_loss = self.train_Qnet(q1,td_target,mini_batch)
                    q2_loss = self.train_Qnet(q2,td_target,mini_batch)
                    pi.train()
                    policy_loss,alpha_loss,alpha_tlogs = self.train_pinet(pi,q1,q2,mini_batch)

                    self.soft_update(q1,q1_target)
                    self.soft_update(q2,q2_target)
                    writer.add_scalar('loss/critic_1', q1_loss, updates)
                    writer.add_scalar('loss/critic_2', q2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', alpha_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha_tlogs, updates)
                    check_policy_loss +=policy_loss
                    check_q1_loss += q1_loss
                    check_q2_loss += q2_loss
                    updates += 1
                logger.debug(check_policy_loss)

            # print(f'{n_epi}episode -> loss:{loss}')
            if n_epi % print_interval == 0 and n_epi != 0:
                episode_reward_lst = np.array(episode_reward_lst)
                writer.add_scalar('reward_mean/episode', episode_reward_lst.mean(), n_epi)
                writer.add_scalar('reward_min/episode', episode_reward_lst.min(), n_epi)
                writer.add_scalar('reward_max/episode', episode_reward_lst.max(), n_epi)
                logger.info("n_episode :{},policy_loss:{:.3f}, q1_loss:{:.3f} ,q2_loss:{:.3f} episode_reward_mean_min_max: ({:.3f})_({:.3f})_({:.3f}), n_buffer : {}".format(
                    n_epi,check_policy_loss,check_q1_loss,check_q2_loss, episode_reward_lst.mean(),episode_reward_lst.min(),episode_reward_lst.max(), memory.size()))
                logger.debug(result_data)
                episode_reward_lst = []

                #tensorboard --logdir=runs
        torch.save(pi, self.model_path)
        writer.close()

    def validate(self,env):
        valid_pi = torch.load(self.model_path)
        valid_pi.mode = 'valid'
        for n_epi in range(1):
            episode_reward = 0.0

            result_dict = {}


            s = env.reset(env.valid_data)

            done = False
            ######
            pf_data_index = 0
            ######

            while not done:
                s = s.reshape([1, -1])
                valid_pi.eval()
                a, _ = valid_pi.forward(s)


                confidence = a.detach().numpy().squeeze() * env.valid_pf_data.iloc[pf_data_index + self.window_size - 1]
                logger.debug(f'ta score->({env.valid_pf_data.iloc[pf_data_index + self.window_size - 1]})')
                logger.debug(f'action->({a.detach().numpy().squeeze()})')
                confidence = np.exp(confidence) / np.sum(np.exp(confidence))
                pf_data_index += 1



                s_prime, r, done, portfolio_dict = env.step(env.valid_data, a, confidence, self.mode)
                result_dict.update(portfolio_dict)
                s = s_prime
                episode_reward += r

                if done:
                    columns = ['Portfolio_value'] + env.train_data_lst
                    result_data = pd.DataFrame.from_dict(result_dict, orient='index')
                    result_data.columns = columns



                    logger.debug(f'valid_reward->({episode_reward})')


        return result_data

    def test(self,env):
        metrics = Metrics()
        test_pi = torch.load(self.model_path)
        test_pi.mode = 'test'

        for n_epi in range(1):
            episode_reward = 0.0

            result_dict = {}

            s = env.reset(env.test_data)

            done = False
            ######
            pf_data_index = 0
            ######
            while not done:
                test_pi.eval()
                a, _ = test_pi.forward(s)
                a = a.detach().numpy().squeeze()


                portfolio_portion = np.exp(a[-3:]) / np.sum(np.exp(a[-3:]))

                # portfolio_portion = np.array([1/3,1/3,1/3])
                # portfolio_portion = np.exp(a) / np.sum(np.exp(a))

                stock_a = np.exp(a[:5]) / np.sum(np.exp(a[:5]))
                crypto_a = np.exp(a[5:10]) / np.sum(np.exp(a[5:10]))

                # stock_a = np.array([1 / 5] * (5))
                # crypto_a = np.array([1 / 5] * (5))

                stock_a = stock_a * portfolio_portion[1]
                crypto_a = crypto_a * portfolio_portion[2]
                confidence = np.concatenate((stock_a, crypto_a))
                confidence = np.append(confidence, portfolio_portion[0])

                ###############
                pf_data_index += 1
                s_prime, r, done, portfolio_dict = env.step(env.test_data, a, confidence, self.mode)
                result_dict.update(portfolio_dict)
                logger.debug(f'portfolio portion: {portfolio_portion}')
                logger.debug(f'{env.test_return_data.index[pf_data_index + self.window_size-1]}:a->{a}')
                logger.debug(f'return_data->{env.test_return_data.iloc[pf_data_index + self.window_size-1]}')

                s = s_prime
                episode_reward += r

                if done:
                    columns = ['Portfolio_value'] + env.train_data_lst
                    result_data = pd.DataFrame.from_dict(result_dict, orient='index')
                    result_data.columns =columns
                    logger.debug(f'Test_reward->({episode_reward})')

        return result_data


class DQN():
    def __init__(self, data_num, model_path, args):
        # for cuda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.data_num = data_num
        self.batch_size = args.batch_size

        self.gamma = args.discount_factor
        self.tau = args.tau
        self.alpha = args.alpha

        self.lr = args.lr
        self.window_size = args.window_size
        self.output_dim = args.output_dim
        self.mode = args.mode
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.updates_per_step = args.updates_per_step

    def train(self, env):
        value_network = Qnet(self.batch_size, env.data_num, output_dim=self.output_dim, lr=self.lr,
                             window_size=self.window_size)
        target_network = Qnet(self.batch_size, env.data_num, output_dim=self.output_dim, lr=self.lr,
                              window_size=self.window_size)
        target_network.load_state_dict(value_network.state_dict())
        memory = ReplayBuffer(self.batch_size, self.data_num)
        print_interval = 50
        loss_sum = 0.0
        episode_reward_lst = []
        for n_epi in range(301):
            result_dict = {}
            result_data = pd.DataFrame()
            epsilon = max(0.01, 0.5 - 0.1 * (n_epi / 100))

            s = env.reset(env.train_data)
            done = False

            ######
            pf_data_index = 0
            ######

            while not done:
                episode_reward = 0.0
                a, confidence = value_network.sample_action(s, epsilon)

                confidence = np.exp(confidence) / np.sum(np.exp(confidence))
                #######################
                cash_x = []
                z = 0
                for i in range(len(confidence)):
                    if i > 9:
                        z += confidence[i]
                    else:
                        cash_x.append(confidence[i])
                cash_x.append(z)
                confidence = cash_x
                pf_data_index += 1
                ###########################
                s_prime, r, done, portfolio_value = env.step(env.train_data, a, confidence, self.mode)
                result_dict.update(portfolio_value)
                done_mask = 0.0 if done else 1.0
                memory.put((s.squeeze(), a, r, s_prime.squeeze(), done_mask))
                s = s_prime
                episode_reward += r
                if done:
                    episode_reward_lst.append(episode_reward)
                    writer.add_scalar('reward per episode', episode_reward, n_epi)
                    columns = ['Portfolio_value'] + env.train_data_lst
                    result_data = pd.DataFrame.from_dict(result_dict, orient='index')
                    result_data.columns = columns

                    break

            if memory.size() > 2000:
                s, a, r, s_prime, done_mask = memory.sample(self.batch_size)
                q_out = value_network(s).reshape(self.batch_size, env.data_num + 5, self.output_dim)
                a = value_network.revert_action(a)
                q_a = torch.gather(q_out, 2, a).squeeze()
                max_q_prime = \
                target_network(s_prime).reshape(self.batch_size, env.data_num + 5, self.output_dim).max(2)[
                    0]  # .to(device)
                target = r + self.gamma * max_q_prime * done_mask
                loss = F.smooth_l1_loss(q_a, target, beta=0.5)  # target.to(device)

                value_network.optimizer.zero_grad()
                loss.backward()
                value_network.optimizer.step()
                loss_sum += loss

            if n_epi % print_interval == 0 and n_epi != 0:
                target_network.load_state_dict(value_network.state_dict())
                episode_reward_lst = np.array(episode_reward_lst)
                writer.add_scalar('reward_mean/episode', episode_reward_lst.mean(), n_epi)
                writer.add_scalar('reward_min/episode', episode_reward_lst.min(), n_epi)
                writer.add_scalar('reward_max/episode', episode_reward_lst.max(), n_epi)
                logger.info(
                    "n_episode :{}, loss :{}, episode_reward_mean_min_max : ({:.4f})_({:.4f})_({:.4f})_, n_buffer : {}, eps : {:.1f}%".format(
                        n_epi, loss_sum / print_interval, episode_reward_lst.mean(), episode_reward_lst.min(),
                        episode_reward_lst.max(), memory.size(), epsilon * 100))
                logger.debug(result_data)
                episode_reward_lst = []
                loss_sum = 0.0

        torch.save(target_network, self.model_path)

    def validate(self, env):
        valid_network = torch.load(self.model_path)

        for n_epi in range(1):
            result_dict = {}
            episode_reward = 0.0
            s = env.reset(env.valid_data)
            done = False
            ######
            pf_data_index = 0
            ######
            while not done:
                a, confidence = valid_network.sample_action(s, 0)
                confidence = np.exp(confidence) / np.sum(np.exp(confidence))
                ####################### cash_adjust############
                cash_x = []
                z = 0
                for i in range(len(confidence)):
                    if i > 9:
                        z += confidence[i]
                    else:
                        cash_x.append(confidence[i])
                cash_x.append(z)
                confidence = cash_x

                logger.debug(f'confidence:{confidence}')
                logger.debug(
                    f'{env.test_return_data.index[pf_data_index + self.window_size - 1]}:confidence->{confidence}')
                logger.debug(
                    f'{env.test_return_data.index[pf_data_index + self.window_size]}:return_data->{env.valid_return_data.iloc[pf_data_index + self.window_size]}')

                pf_data_index += 1

                s_prime, r, done, portfolio_value = env.step(env.valid_data, a, confidence, self.mode)
                result_dict.update(portfolio_value)
                s = s_prime
                episode_reward += r
                if done:
                    columns = ['Portfolio_value'] + env.train_data_lst
                    result_data = pd.DataFrame.from_dict(result_dict, orient='index')
                    result_data.columns = columns
                    cm_return = (result_data['Portfolio_value'] + 1).cumprod() - 1
                    logger.debug(f'test_reward->({episode_reward})')
                    break

        return result_data

    def test(self, env):
        test_network = torch.load(self.model_path)

        episode_reward = 0.0

        for n_epi in range(1):
            result_dict = {}
            s = env.reset(env.test_data)
            done = False
            ######
            pf_data_index = 0
            ######

            while not done:
                a, confidence = test_network.sample_action(s, 0)
                confidence = np.exp(confidence) / np.sum(np.exp(confidence))
                ####################### cash_adjust############
                cash_x = []
                z = 0
                for i in range(len(confidence)):
                    if i > 9:
                        z += confidence[i]
                    else:
                        cash_x.append(confidence[i])
                cash_x.append(z)
                confidence = cash_x

                logger.debug(f'confidence:{confidence}')
                logger.debug(
                    f'{env.test_return_data.index[pf_data_index + self.window_size - 1]}:confidence->{confidence}')
                logger.debug(
                    f'{env.test_return_data.index[pf_data_index + self.window_size]}:return_data->{env.test_return_data.iloc[pf_data_index + self.window_size]}')

                pf_data_index += 1

                s_prime, r, done, portfolio_value = env.step(env.test_data, a, confidence, self.mode)
                result_dict.update(portfolio_value)
                s = s_prime
                episode_reward += r
                if done:
                    columns = ['Portfolio_value'] + env.train_data_lst
                    result_data = pd.DataFrame.from_dict(result_dict, orient='index')
                    result_data.columns = columns
                    cm_return = (result_data['Portfolio_value'] + 1).cumprod() - 1
                    logger.debug(f'test_reward->({episode_reward})')
                    break

            return result_data




