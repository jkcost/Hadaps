import os
import pandas as pd
import numpy as np
import re
import random
import torch
import torch.nn.functional as F


def convert_label(x):
    if x > 0.03:
        x = 1.0
    elif x < -0.03:
        x = -1.0
    else:
        x = 0.0
    return x


def convert_variation(x):
    if x < 0:
        x = -1
    else:
        x = 1
    return x


class Environment:

    def __init__(self, start_date, train_date, valid_date, test_date, market_type, use_ta, ta_parameter, window_size,
                 rl_method, include_cash):

        self.start_date = start_date
        self.train_date = train_date
        self.valid_date = valid_date
        self.test_date = test_date
        self.market_type = market_type

        self.use_ta = use_ta
        self.ta_parameter = ta_parameter
        self.window_size = window_size
        self.rl_method = rl_method
        self.include_cash = include_cash

        self.cal_portfolio_lst = [0.0]
        self.index = 0
        self.chart_data, self.training_data, self.return_data, self.pf_data, self.data_num, self.stock_num, self.market_index, self.train_data_lst = self.load_data(
            self.test_date, market_type=self.market_type)
        self.train_chart_data, self.valid_chart_data, self.test_chart_data, self.train_return_data, self.valid_return_data, self.test_return_data, self.train_data, self.valid_data, self.test_data, self.train_pf_data, self.valid_pf_data, self.test_pf_data = self.split_data()


    def reset(self, data):
        self.index = 0
        self.cal_portfolio_lst = [0.0]

        s = torch.from_numpy(np.array(data.iloc[self.index:self.index + self.window_size])).float()
        s = s.split(self.data_num, dim=1)
        s = torch.stack([s[0].reshape([1, -1]), s[1].reshape([1, -1])], dim=1)
        return s

    def step(self, data, action, confidence, mode):
        done = False
        self.index += 1

        s_prime = torch.from_numpy(np.array(data.iloc[self.index:self.index + self.window_size])).float()
        s_prime = s_prime.split(self.data_num, dim=1)
        s_prime = torch.stack([s_prime[0].reshape([1, -1]), s_prime[1].reshape([1, -1])], dim=1)

        if mode == 'train':
            portfolio_dict, portfolio_value = self.cal_portfolio(self.train_return_data, self.train_pf_data, confidence)
            self.cal_portfolio_lst.append(portfolio_value)
            r = self.cal_reward(self.train_return_data, action, self.rl_method)
            if self.index == (len(self.train_data) - self.window_size):
                done = True

        elif mode == 'valid':
            portfolio_dict, portfolio_value = self.cal_portfolio(self.valid_return_data, self.train_pf_data, confidence)
            self.cal_portfolio_lst.append(portfolio_value)
            r = self.cal_reward(self.valid_return_data, action, self.rl_method)
            if self.index == (len(self.valid_data) - self.window_size):
                done = True

        elif mode == 'test':
            portfolio_dict, portfolio_value = self.cal_portfolio(self.test_return_data, self.train_pf_data, confidence)
            self.cal_portfolio_lst.append(portfolio_value)
            r = self.cal_reward(self.test_return_data, action, self.rl_method)
            if self.index == (len(self.test_data) - self.window_size):
                done = True
        else:
            pass

        return s_prime, r, done, portfolio_dict

    def cal_reward(self, return_data, action, rl_method):
        if rl_method == 'sac':
            reward = self.cal_portfolio_lst[self.index]

        else:
            reward = action * np.array(return_data.iloc[self.index + self.window_size - 1])
            reward = reward.sum()
        return reward

    def cal_portfolio(self, return_data, pf_data, confidence):

        portfolio_value = (confidence * np.array(return_data.iloc[self.index + self.window_size - 1])).sum()
        portfolio_info = {
            f'{return_data.index[self.index + self.window_size - 1].strftime("%Y-%m-%d")}': [portfolio_value] + [i for i in confidence]}
        return portfolio_info, portfolio_value

    def load_data(self, test_date=None, market_type=None):
        train_data_lst = []
        df_last_date = self.start_date
        stock_list = os.listdir(f'./datasets/nasdaq')
        crypto_list = os.listdir(f'./datasets/crypto')
        stock_num = len(os.listdir(f'./datasets/nasdaq')) - 1
        if market_type == 'crypto':
            stock_list = os.listdir(f'./datasets/{market_type}')
            chart_data = pd.read_csv(f'./datasets/{market_type}/btc.csv')
            chart_data = chart_data.sort_values(by='Date').reset_index(drop=True)
            chart_data.index = pd.to_datetime(chart_data['Date'], format='%Y-%m-%d', errors='ignore')
            chart_data['Date'] = chart_data['Date'].str.replace('-', '')
            chart_data['Close'] = chart_data['Close'].str.replace(',', '').astype('float')
            market_index = 'BTC-USD'
        p = re.compile('^\^')
        for z, crypto in enumerate(crypto_list):
            for i, stock in enumerate(stock_list):
                if p.match(stock):
                    chart_data = pd.read_csv(f'./datasets/nasdaq/{stock}')
                    chart_data = chart_data.sort_values(by='Date').reset_index(drop=True)
                    chart_data.index = pd.to_datetime(chart_data['Date'], format='%Y-%m-%d', errors='ignore')
                    chart_data['Date'] = chart_data['Date'].str.replace('-', '')

                    market_index = stock.split('.')[0]

                else:
                    df = pd.read_csv(f'./datasets/{market_type}/{stock}', thousands=',',
                                     converters={'Date': lambda x: str(x)})
                    df = df.sort_values(by='Date').reset_index(drop=True)
                    df['Date'] = df['Date'].str.replace('-', '')
                    df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='ignore')
                    if df['Date'][0] >= df_last_date:
                        df_last_date = df['Date'][0]
            df = pd.read_csv(f'./datasets/crypto/{crypto}', thousands=',',
                             converters={'Date': lambda x: str(x)})
            df = df.sort_values(by='Date').reset_index(drop=True)
            df['Date'] = df['Date'].str.replace('-', '')
            df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='ignore')
            if df['Date'][0] >= df_last_date:
                df_last_date = df['Date'][0]

        chart_data = chart_data[(chart_data['Date'] >= df_last_date) & (
                chart_data['Date'] <= test_date)]  # to match date with training_data

        pf_data = pd.DataFrame(index=chart_data.index.copy())
        training_data = pd.DataFrame(index=chart_data.index.copy())
        return_data = pd.DataFrame(index=chart_data.index.copy())

        if self.market_type == 'nasdaq':
            asset_list = stock_list
        elif self.market_type == 'crypto':
            asset_list = crypto_list
        else:
            asset_list = stock_list + crypto_list

        for i, asset in enumerate(asset_list):
            if p.match(asset):
                asset = asset.split('.')[0]
                index_name = asset
                train_data_lst.append(asset)
                continue

            df = pd.read_csv(f'./datasets/both/{asset}', thousands=',',
                             converters={'Date': lambda x: str(x)})
            df = df.sort_values(by='Date').reset_index(drop=True)

            df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='ignore')

            df = df.loc[chart_data.index]
            df['Date'] = df['Date'].str.replace('-', '')
            df = df[(df['Date'] >= df_last_date) & (df['Date'] <= test_date)]
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            df = df[['Date', 'Close']]
            asset = asset.split('.')[0]
            train_data_lst.append(asset)

            df[f'{train_data_lst[i]}_Return'] = df['Close'].pct_change(1)
            # df[f'{train_data_lst[i]}_Close'] = (df['Close'] - np.mean(df['Close']))/ np.std(df['Close'],ddof=1)
            df[f'{train_data_lst[i]}_Close'] = df['Close']

            ##### Portfolio variation ####
            df[f'{train_data_lst[i]}_Variation_impact'] = df[f'{train_data_lst[i]}_Return'].apply(convert_variation)
            df[f'{train_data_lst[i]}_Variation'] = df[f'{train_data_lst[i]}_Close'].rolling(self.window_size).std(
                ddof=1) * df[f'{train_data_lst[i]}_Variation_impact']
            pf_data = pd.concat([pf_data, df[f'{train_data_lst[i]}_Variation'].fillna(0)], axis=1)
            return_data = pd.concat([return_data, df[f'{train_data_lst[i]}_Return']], axis=1)
            training_data = pd.concat([training_data, df[f'{train_data_lst[i]}_Close']], axis=1)


        if self.market_type != 'crypto':
            train_data_lst.remove(index_name)
        df['Cash_Return'] = pd.DataFrame(data=np.zeros(len(return_data.index)), index=return_data.index)
        return_data = pd.concat([return_data, df['Cash_Return']], axis=1)
        data_num = len(train_data_lst)
        train_data_lst.append('cash')
        ###### variation###
        training_data = pd.concat([training_data, pf_data], axis=1)

        # if self.market_type == 'both':
        #     training_data = training_data[['AAPL_Close', 'AMZN_Close', 'GOOGL_Close', 'META_Close', 'MSFT_Close',
        #                                    'ada_Close', 'bnb_Close', 'btc_Close', 'eth_Close', 'xrp_Close',
        #                                    'AAPL_Variation', 'AMZN_Variation', 'GOOGL_Variation', 'META_Variation',
        #                                    'MSFT_Variation', 'ada_Variation', 'bnb_Variation', 'btc_Variation',
        #                                    'eth_Variation', 'xrp_Variation']]
        #
        #     return_data = return_data[['AAPL_Return', 'AMZN_Return', 'GOOGL_Return', 'META_Return', 'MSFT_Return',
        #                                'ada_Return', 'bnb_Return', 'btc_Return', 'eth_Return', 'xrp_Return',
        #                                'Cash_Return']]
        #     train_data_lst = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'ada', 'bnb', 'btc', 'eth', 'xrp', 'cash']

        return chart_data, training_data, return_data, pf_data, data_num, stock_num, market_index, train_data_lst

    def split_data(self):
        train_chart_data = self.chart_data[(self.chart_data.index <= self.train_date)]
        valid_chart_data = self.chart_data[
            (self.chart_data.index > self.train_date) & (self.chart_data.index <= self.valid_date)]
        test_chart_data = self.chart_data[(self.chart_data.index > self.valid_date)]

        train_return_data = self.return_data[(self.return_data.index <= self.train_date)]
        valid_return_data = self.return_data[
            (self.return_data.index > self.train_date) & (self.chart_data.index <= self.valid_date)]
        test_return_data = self.return_data[(self.return_data.index > self.valid_date)]

        train_data = self.training_data[(self.training_data.index <= self.train_date)]
        train_static = self.training_data[(self.training_data.index <= self.train_date)]
        train_data = (train_data - np.mean(train_static)) / np.std(train_static, ddof=1)
        valid_data = self.training_data[
            (self.training_data.index > self.train_date) & (self.chart_data.index <= self.valid_date)]
        valid_data = (valid_data - np.mean(train_static)) / np.std(train_static, ddof=1)
        test_data = self.training_data[(self.training_data.index > self.valid_date)]
        test_data = (test_data - np.mean(train_static)) / np.std(train_static, ddof=1)

        train_pf_data = self.pf_data[(self.pf_data.index <= self.train_date)]
        valid_pf_data = self.pf_data[(self.pf_data.index > self.train_date) & (self.pf_data.index <= self.valid_date)]
        test_pf_data = self.pf_data[(self.pf_data.index > self.valid_date)]

        return train_chart_data, valid_chart_data, test_chart_data, train_return_data, valid_return_data, test_return_data, train_data, valid_data, test_data, train_pf_data, valid_pf_data, test_pf_data


