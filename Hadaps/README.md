# Hadaps: Hierarchical Adaptive Multi-Asset Portfolio Selection

Multi-asset portfolio selection is an asset allocation strategy involving a variety of assets. Adaptive investment strategies which consider the dynamic market characteristics of individual assets and asset classes are vital for maximizing returns and minimizing risks. We introduce HADAPS, a novel computational method for multi-asset portfolio selection which utilizes the Soft-Actor-Critic (SAC) framework enhanced with Hierarchical Policy Network. Contrary to previous approaches that have relied on heuristics for constructing asset allocations,  HADAPS directly outputs a continuous vector of action values depending on current market conditions. In addition, HADAPS performs multi-asset portfolio selection involving multiple asset classes. 


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
|                 stocks_data.csv                  |        [num_days, num_features]   |       the inputs for stock asset portolio        |
|                 crypto_data.csv                  |       [num_days, num_features]       |     the inputs for crypto asset portfolio      |



These files should be placed in the ./datasets/*(asset classes) folder, e.g. ./datasets/crypto/btc.csv
also all input data and index data must be included in the ./datasets/both folder

## Training

As an example, after putting data source file to the data folder, you can simply run:

`python run.py `

Some of the available arguments are:

| Argument          | Description                                                | Default                     | Type  |
| ----------------- | ---------------------------------------------------------- | --------------------------- | ----- |
| `--rl_method`        | Select to rl_method                                 | SAC                | str   |
| `--window_size`    | Input window size                                          | 5 (days)                | int   |
| `--discount_factor`        | Dis count factor for reward                                               | 0.9                      | float   |
| `--ta_parameter`             | The parameter for adaptive portpolio selection | 5                | int   |
| `--batch_size`    | Batch size number                                          | 32                          | Int   |
| `--lr`            | learning rate                                              | 1e-5                        | float |
| `--tau`         | target smoothing coefficient(τ)         | 0.005                        | float |
| `--alpha`    | Temperature parameter α determines the relative importance of the entropy  term against the reward     | 0.2       | float  |
| `--automatic_entropy_tuning`        | Automaically adjust α                       | False                       | bool  |




