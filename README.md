# Hadaps: Hierarchical Adaptive Multi-Asset Portfolio Selection

Multi-asset portfolio selection is an asset allocation strategy involving a variety of assets. Adaptive investment strategies which consider the dynamic market characteristics of individual assets and asset classes are vital for maximizing returns and minimizing risks. We introduce \model, a novel computational method for multi-asset portfolio selection which utilizes the Soft-Actor-Critic (SAC) framework enhanced with Hierarchical Policy Network. Contrary to previous approaches that have relied on heuristics for constructing asset allocations,  \model~\textit{directly} outputs a continuous vector of action values depending on current market conditions. In addition, \model~performs multi-asset portfolio selection involving \textit{multiple} asset classes. 
Experimental results show that \model~ outperforms baseline approaches in not only cumulative returns but also risk-adjusted metrics. These results are based on market price data from sectors with various behavioral characteristics.
% Experimental results show that \model~outperforms baseline approaches in not only cumulative returns but also risk-adjusted metrics which were based market price data consisting three asset classes and eleven assets. 
Furthermore, qualitative analysis shows \model' ability to adaptively shift portfolio selection strategies in dynamic market conditions where asset classes and different assets are uncorrelated to each other.~\footnote{The code and data will be publicly released upon paper acceptance.}

## Content

1. [Requirements](#Requirements)
2. [Data Preparing]()
3. [Training](Training)




## Requirements

- Python 3.6 or higher.
- torch >== 1.12.1.
- Pandas >= 0.25.1
- Numpy >= 1.18.1
- TensorFlow >= 1.14.0 (For you can easyly use TensorBoard)
- ...

## Data Preparing


The following files are needed:

|                    File_name                     |                  shape                   |                  description                   |
| :----------------------------------------------: | :--------------------------------------: | :--------------------------------------------: |
|                 stocks_data.npy                  | [num_stocks, num_days, num_ASU_features] |       the inputs for asset scoring unit        |
|                 market_data.npy                  |       [num_days, num_MSU_features]       |     the inputs for marketing scoring unit      |
|                     ror.npy                      |          [num_stocks, num_days]          | rate of return file for calculating the return |
| relation_file (e.g. industry_classification.npy) |         [num_stocks, num_stocks]         |     the relation matrix used in GCN layer      |



These files should be placed in the ./data/INDEX_NAME folder, e.g. ./data/DJIA/stocks_data.npy

## Training

As an example, after putting data source file to the data folder, you can simply run:

`python run.py -c hyper.json`

Some of the available arguments are:

| Argument          | Description                                                | Default                     | Type  |
| ----------------- | ---------------------------------------------------------- | --------------------------- | ----- |
| `--rl_method`        | Deafult configuration file                                 | hyper.json                  | str   |
| `--window_size`    | Input window size                                          | 13 (weeks)                  | int   |
| `--discount_factor`        | Stock market                                               | DJIA                        | str   |
| `--ta_parameter`             | The number of stocks participating in long/short each time | 4 (for DJIA)                | int   |
| `--batch_size`    | Batch size number                                          | 37                          | Int   |
| `--lr`            | learning rate                                              | 1e-6                        | float |
| `--automatic_entropy_tuning`         | Coefficient for adjusting lr between ASU and MSU           | 0.05                        | float |
| `--tau`    | Whether to use spatial attention and GCN layer in ASU      | True                        | bool  |
| `--alpha`        | Whether to use market scoring unit                         | True                        | bool  |




