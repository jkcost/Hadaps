import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import logging
import json
import argparse



import pandas as pd
import torch
from environment import Environment

from model import DQN,SAC,Heuristic
from visualizer import Visualizer
from metrics import Metrics

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(1)}"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_lst = [4,45,35,15,79,1992,12,1,1996,4,19]

seed_everything(seed_lst[0])






path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.join(path,'experiment')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
list_str  = '[train,valid,test]'


parser = argparse.ArgumentParser()
parser.add_argument('--title',default='Hadaps')
parser.add_argument('--mode',choices=['train','valid','test'],default='train')
parser.add_argument('--rl_method', choices=['dqn','sac','naive'],default='sac')
parser.add_argument('--lr',type=float,default=0.0001)
parser.add_argument('--discount_factor',type=float,default=0.99)

parser.add_argument('--name',default=datetime.datetime.today().strftime(('%Y%m%d')))
parser.add_argument('--start_date',default='20080331')
parser.add_argument('--train_date', default='20201231')
parser.add_argument('--valid_date',default='20201231')
parser.add_argument('--test_date',default='20210131')
parser.add_argument('--output_dim',type=int, default=3)

parser.add_argument('--ta_parameter',choices=[1,3,5,7,9],type=int,default=1)
parser.add_argument('--window_size',type=int,default=5)
parser.add_argument('--batch_size',choices=[8,16,32,64,128,256],default=32)

parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')

parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')

parser.add_argument('--updates_per_step', type=int, default=10, metavar='N',
                    help='model updates per simulator step (default: 10)')

args = parser.parse_args()

output_name = f'{args.mode}_{args.name}_{args.rl_method}_{args.train_date}_{args.valid_date}_{args.window_size}_{args.net}_{args.lr}_{args.discount_factor}_{args.use_ta}_{args.ta_parameter}_{args.include_cash}'


#to make output_path
output_path = os.path.join(base_path,f'{args.title}',f'{args.mode}',f'{args.rl_method}',output_name)
if not os.path.isdir(output_path):
    os.makedirs(output_path)

graph_path = os.path.join(output_path,'graph')
if not os.path.isdir(graph_path):
    os.makedirs(graph_path)

params = json.dumps(vars(args))
with open(os.path.join(output_path,'params.json'),'w') as f:
    f.write(params)

model_dir =  os.path.join(path,f'models')
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
#

model_path_both = os.path.join(model_dir, f'{args.rl_method}_both_{args.valid_date}_{args.lr}_{args.discount_factor}_{args.use_ta}_{args.ta_parameter}_{args.include_cash}.h5')
model_path_stock = os.path.join(model_dir, f'{args.rl_method}_nasdaq_{args.valid_date}_{args.lr}_{args.discount_factor}_{args.use_ta}_{args.ta_parameter}_{args.include_cash}.h5')
model_path_crypto = os.path.join(model_dir, f'{args.rl_method}_crypto_{args.valid_date}_{args.lr}_{args.discount_factor}_{args.use_ta}_{args.ta_parameter}_{args.include_cash}.h5')

log_path = os.path.join(output_path, f'{output_name}.log')
if os.path.exists(log_path):
    os.remove(log_path)

logging.basicConfig(format='%(message)s')
logger = logging.getLogger('Portfolio')
logger.setLevel(logging.DEBUG)
logger.propagate = False
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.info(params)



both_env = Environment(start_date = args.start_date,train_date=args.train_date, valid_date=args.valid_date, test_date=args.test_date,
                       market_type='both',use_ta=args.use_ta,ta_parameter=args.ta_parameter, window_size=args.window_size, rl_method=args.rl_method,include_cash=args.include_cash)
stock_env = Environment(start_date = args.start_date,train_date=args.train_date, valid_date=args.valid_date, test_date=args.test_date,
                        market_type='nasdaq',use_ta=args.use_ta,ta_parameter=args.ta_parameter, window_size=args.window_size, rl_method=args.rl_method,include_cash=args.include_cash)
crypto_env = Environment(start_date = args.start_date,train_date=args.train_date, valid_date=args.valid_date, test_date=args.test_date,
                         market_type='crypto',use_ta=args.use_ta,ta_parameter=args.ta_parameter, window_size=args.window_size, rl_method=args.rl_method,include_cash=args.include_cash)

metrics = Metrics()

def train():
    logger.info(f"Train period {both_env.train_chart_data['Date'].iloc[0]} ~ {both_env.train_chart_data['Date'].iloc[-1]}")
    if args.rl_method =='dqn':
        logger.info(f"Training start for both markets")
        dqn_model_both = DQN(both_env.data_num,model_path_both,args)
        dqn_model_both.train(both_env)


    elif args.rl_method =='sac':
        logger.info(f"Training start for both markets")
        sac_model_both = SAC(both_env.data_num,output_path,model_path_both,args)
        sac_model_both.train(both_env)

def valid():
    visualizer =Visualizer(both_env,args.mode)
    logger.info(f"valid period {both_env.valid_chart_data['Date'].iloc[0]} ~ {both_env.valid_chart_data['Date'].iloc[-1]}")
    if args.rl_method == 'dqn':
        logger.info(f"Validate start for both markets")
        dqn_model_both = DQN(both_env.data_num, model_path_both, args)
        crypto_stock_both = dqn_model_both.validate(both_env)
        crypto_stock_both.to_csv(f'{output_path}/both_markets.csv', encoding="utf-8-sig")
        crypto_stock_both_graph = f'{graph_path}/both_marekets'
        visualizer.confidence(both_env, crypto_stock_both, crypto_stock_both_graph, True)




    elif args.rl_method == 'sac':
        logger.info(f"Validate start for both markets")
        sac_model_both = SAC(both_env.data_num,output_path, model_path_both, args)
        crypto_stock_both=sac_model_both.validate(both_env)
        crypto_stock_both.to_csv(f'{output_path}/both_markets.csv',encoding="utf-8-sig")
        crypto_stock_both_graph = f'{graph_path}/both_marekets'
        visualizer.confidence(both_env,crypto_stock_both,crypto_stock_both_graph,True)


def test():
    visualizer = Visualizer(both_env, args.mode)
    logger.info(f"test period {both_env.test_chart_data['Date'].iloc[0]} ~ {both_env.test_chart_data['Date'].iloc[-1]}")
    if args.rl_method == 'dqn':
        logger.info(f"Test start for both markets")
        dqn_model_both = DQN(both_env.data_num, model_path_both, args)
        crypto_stock_both = dqn_model_both.test(both_env)
        crypto_stock_both.to_csv(f'{output_path}/both_markets.csv', encoding="utf-8-sig")
        crypto_stock_both_graph = f'{graph_path}/both_markets'
        visualizer.confidence(both_env, crypto_stock_both, crypto_stock_both_graph, True)


        heuristic_model = Heuristic(both_env, args)
        monkey_invest = heuristic_model.monkey_invest()
        fair_invest = heuristic_model.fair_invest()

        stock_result = pd.concat([crypto_stock_both['Portfolio_value'], monkey_invest, fair_invest,
                                  ], axis=1)
        stock_result.columns = ['crypto_stock_both', 'monkey_invest', 'fair_invest']


        metrics_result = metrics.metrics(stock_result)
        metrics_result.to_csv(f'{output_path}/total_metrics.csv', encoding="utf-8-sig")
        visualizer.portfolio(metrics.compsum(stock_result), graph_path)


        logger.info(f'learning_Rate :{args.lr}, discount factor : {args.discount_factor}')



    elif args.rl_method == 'sac':
        logger.info(f"Test start for both markets")
        sac_model_both = SAC(both_env.data_num,output_path, model_path_both, args)
        crypto_stock_both = sac_model_both.test(both_env)
        crypto_stock_result= pd.concat([crypto_stock_both,both_env.test_return_data[args.window_size+1:]],axis=1)
        crypto_stock_result.to_csv(f'{output_path}/both_markets.csv', encoding="utf-8-sig")
        crypto_stock_both_graph = f'{graph_path}/both_marekets'
        visualizer.confidence(both_env,crypto_stock_both, crypto_stock_both_graph, True)

        heuristic_model = Heuristic(both_env, args)
        monkey_invest = heuristic_model.monkey_invest()
        fair_invest = heuristic_model.fair_invest()


        #############################################
        stock_result = pd.concat([crypto_stock_both['Portfolio_value'], monkey_invest, fair_invest
                                ], axis=1)
        stock_result.columns = ['crypto_stock_both', 'monkey_invest', 'fair_invest']

        metrics_result = metrics.metrics(stock_result)

        logger.info(f'learning_Rate :{args.lr}, discount factor : {args.discount_factor}')


        metrics_result.to_csv(f'{output_path}/total_metrics.csv', encoding="utf-8-sig")
        visualizer.portfolio(metrics.compsum(stock_result), graph_path)


if __name__ =='__main__':
    if args.mode == 'train':
        train()
    elif args.mode =='valid':
        valid()
    else:
        test()
