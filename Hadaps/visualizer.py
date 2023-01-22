import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import (
    FormatStrFormatter as _FormatStrFormatter,
    FuncFormatter as _FuncFormatter
)
class Visualizer:
    def __init__(self,env,mode):
        self.env = env
        self.mode = mode

    def confidence(self,env,df,graph_path,ranking=False):
        df.index = pd.to_datetime(df.index,format='%Y-%m-%d',errors='ignore')
        if ranking ==True:
            name_list =['top5','bottom5']
            bottom5 = df.loc[df['Portfolio_value'].sort_values().head().index]
            top5 = df.loc[df['Portfolio_value'].sort_values().tail().index]
            for i in name_list:
                if i =='top5':
                    df = top5
                else:
                    df = bottom5

                labels = list(df.index.strftime('%Y%m%d'))
                data = np.array(df.iloc[:, 1:])

                data_cum = data.cumsum(axis=1)
                category_colors = plt.get_cmap('RdYlGn')(
                    np.linspace(0.15, 0.85, data.shape[1]))

                fig, ax = plt.subplots(figsize=(15, 25))
                ax.invert_yaxis()
                ax.xaxis.set_visible(False)
                ax.set_xlim(0, np.sum(data, axis=1).max())

                for z, (colname, color) in enumerate(zip(env.train_data_lst, category_colors)):
                    widths = data[:, z]
                    starts = data_cum[:, z] - widths
                    ax.barh(labels, widths, left=starts, height=0.8,
                            label=colname, color=color)
                    xcenters = starts + widths / 2

                    r, g, b, _ = color
                    text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
                    for y, (x, c) in enumerate(zip(xcenters, widths)):
                        ax.text(x, y, str(round(c * 100, 1)) + '%', ha='center', va='center',
                                color=text_color, fontsize='x-small')
                ax.legend(ncol=len(env.train_data_lst), bbox_to_anchor=(-0.16, 0.95),
                          loc='lower left', fontsize='medium')
                if i =='top5':
                    ax.set_title('Portoflio_return top5')
                else:
                    ax.set_title('Portoflio_return bottom5')
                plt.savefig(f'{graph_path}_{i}.png')
                # plt.show()



        else:
            labels = list(df.index)
            data = np.array(df.iloc[:, 1:])

            data_cum = data.cumsum(axis=1)
            category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, data.shape[1]))

            fig, ax = plt.subplots(figsize=(15, 25))
            ax.invert_yaxis()
            ax.xaxis.set_visible(False)
            ax.set_xlim(0, np.sum(data, axis=1).max())

            for i, (colname, color) in enumerate(zip(env.train_data_lst, category_colors)):
                widths = data[:, i]
                starts = data_cum[:, i] - widths
                ax.barh(labels, widths, left=starts, height=0.8,
                        label=colname, color=color)
                xcenters = starts + widths / 2

                r, g, b, _ = color
                text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
                for y, (x, c) in enumerate(zip(xcenters, widths)):
                    ax.text(x, y, str(round(c * 100, 1)) + '%', ha='center', va='center',
                            color=text_color, fontsize='x-small')
            ax.legend(ncol=len(env.train_data_lst), bbox_to_anchor=(-0.15, 0.95),
                      loc='lower left', fontsize='medium')
            plt.savefig(f'{graph_path}')
            # plt.show()



    def portfolio(self,df1,df2,graph_path):
        df1.index = pd.to_datetime(df1.index, format='%Y-%m-%d', errors='ignore')
        df2.index = pd.to_datetime(df2.index, format='%Y-%m-%d', errors='ignore')
        fig, ax = plt.subplots(facecolor='w', sharex=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        fig.suptitle('Validation result' + "\n", y=.99, fontweight="bold",
                     fontsize=14, color="black")
        ax.set_title("\n%s - %s                  " % (
            df1.index.date[:1][0].strftime('%e %b \'%y'),
            df1.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')
        for i in range(len(df1.columns)):
            ax.plot(df1.iloc[:,i],  label=df1.columns[i])
        for i in range(len(df2.columns)):
            ax.plot(df2.iloc[:,i],  label=df2.columns[i])
        ax.legend(fontsize=12)
        ax.yaxis.set_major_formatter(_FuncFormatter(self.format_pct_axis))
        ax.axhline(0, ls="-", lw=1,
                   color='gray', zorder=1)
        ax.axhline(0, ls="--", lw=1,
                   color='black', zorder=2)
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.savefig(f'{graph_path}/{self.mode}.png')
        # plt.show()

        # ax1.set_ylabel(f'{self.env.market_type}')
        # ax2.set_ylabel(f'{self.mode} -> Portfolio value')
        # ax1.plot(chart_data.index[window_size+1:],
        #          chart_data['Close'].iloc[window_size:].pct_change().iloc[1:])
        # plt.xticks(rotation=45)
        # df.index = pd.to_datetime(df.index)
        # ax2.plot(chart_data.index[window_size+1:], df['Portfolio_value'].iloc[1:])
        # qs.extend_pandas()
        # returns = pd.Series(df['Portfolio_value'])
        # cm_return = (returns +1).cumprod() -1
        # print('Cumulative_return:',cm_return)
        # qs.reports.full(returns,f'{self.env.market_index}')
        # qs.reports.metrics(returns, f'{self.env.market_index}')
        # qs.reports.metrics(returns, 'BTC-USD')



    def format_pct_axis(self,x, _):
        x *= 100  # lambda x, loc: "{:,}%".format(int(x * 100))
        if x >= 1e12:
            res = '%1.1fT%%' % (x * 1e-12)
            return res.replace('.0T%', 'T%')
        if x >= 1e9:
            res = '%1.1fB%%' % (x * 1e-9)
            return res.replace('.0B%', 'B%')
        if x >= 1e6:
            res = '%1.1fM%%' % (x * 1e-6)
            return res.replace('.0M%', 'M%')
        if x >= 1e3:
            res = '%1.1fK%%' % (x * 1e-3)
            return res.replace('.0K%', 'K%')
        res = '%1.0f%%' % x
        return res.replace('.0%', '%')
#
#
#
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, facecolor='w', sharex=True)
#                 ax1.set_ylabel(f'{env.market_type}')
#                 ax2.set_ylabel(f'episode({n_epi}) -> Portfolio value')
#                 ax1.plot(env.train_chart_data.index[self.window_size:],
#                          env.train_chart_data['Close'].iloc[self.window_size:])
#                 plt.xticks(rotation=45)
#                 ax2.plot(env.train_chart_data.index[self.window_size:], result_data['Portfolio_value'])
                # plt.savefig(f'{graph_path}/episode({n_epi}).png')