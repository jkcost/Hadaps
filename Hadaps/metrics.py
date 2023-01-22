import pandas as pd
import numpy as np

class Metrics():

    def metrics(self,returns,rf=0.):

        # win_year, _ = _get_trading_periods(periods_per_year)
        if isinstance(returns,pd.Series):
            df = pd.DataFrame({"returns": returns})
            df = df.fillna(0)
        else:
            df = returns
        # pct multiplier
        pct = 100


        result_df = pd.DataFrame()
        # return df
        for i,name in enumerate(df.columns):
            metrics = pd.DataFrame()


            s_start = {f'{name}': df.iloc[:,i].index.strftime('%Y-%m-%d')[0]}
            s_end = {f'{name}': df.iloc[:,i].index.strftime('%Y-%m-%d')[-1]}
            s_rf = {f'{name}': rf}
            metrics['Start Period'] = pd.Series(s_start)
            metrics['End Period'] = pd.Series(s_end)
            metrics['Risk-Free Rate %'] = pd.Series(s_rf) * 100

            metrics['Cumulative Return %'] = np.around(((df.iloc[:,i].add(1).prod() - 1) * pct),4)
            metrics['Total Return %'] = np.around((df.iloc[:,i].sum() * pct),4)


            metrics['CAGR%'] = self.cagr(df.iloc[:,i]) * pct
            metrics['Sharpe'] = self.sharpe(df.iloc[:,i], rf)
            metrics['Sortino'] = self.sortino(df.iloc[:,i], rf)
            metrics['Omega'] = self.omega(df.iloc[:,i], rf, 0.)



            metrics['Best Day %'] = self.best(df.iloc[:,i]) * pct
            metrics['Worst Day %'] = self.worst(df.iloc[:,i]) * pct
            metrics['Best Month %'] = self.best(df.iloc[:,i], aggregate='M') * pct
            metrics['Worst Month %'] = self.worst(df.iloc[:,i], aggregate='M') * pct
            metrics['Best Year %'] = self.best(df.iloc[:,i], aggregate='A') * pct
            metrics['Worst Year %'] = self.worst(df.iloc[:,i], aggregate='A') * pct
            dd = self.maximum_dd(df.iloc[:, i],name)
            result =pd.concat([metrics,dd],axis=1)
            result_df = pd.concat([result_df,result])


        return result_df
    def cagr(self,returns,rf=0.):
        """
        Calculates the communicative annualized growth return
        (CAGR%) of access returns

        If rf is non-zero, you must specify periods.
        In this case, rf is assumed to be expressed in yearly (annualized) terms
        """
        total = (returns.add(1).prod()-1)

        years = (returns.index[-1] - returns.index[0]).days / 365.
        res = abs(total +1.0)**(1.0 / years) -1
        return res

    def sharpe(self,returns,rf=0.,periods=252,annualize=True):
        """
        Calculates the sharpe ratio of access returns

        If rf is non-zero, you must specify periods.
        In this case, rf is assumed to be expressed in yearly (annualized) terms

        Args:
            * returns (Series, DataFrame): Input return series
            * rf (float): Risk-free rate expressed as a yearly (annualized) return
            * periods (int): Freq. of returns (252/365 for daily, 12 for monthly)
            * annualize: return annualize sharpe?
            * smart: return smart sharpe ratio
        """
        if rf !=0 and periods is None:
            raise Exception('Must provide periods if rf !=0')

        divisor = returns.std(ddof=1)
        res = returns.mean() / divisor

        if annualize:
            return res*np.sqrt(1 if periods is None else periods)

        return res

    def sortino(self,returns,rf=0.,periods=252,annualize=True):
        """
        Calculates the sortino ratio of access returns

        If rf is non-zero, you must specify periods.
        In this case, rf is assumed to be expressed in yearly (annualized) terms

        Calculation is based on this paper by Red Rock Capital
        http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
        """
        if rf != 0 and periods is None:
            raise Exception('Must provide periods if rf != 0')

        downside = np.sqrt((returns[returns <0]**2).sum() / len(returns))
        res = returns.mean() /downside

        if annualize:
            return res * np.sqrt(1 if periods is None else periods)

        return res

    def omega(self,returns,rf=0.0, required_return=0.0, periods=252):
        """
        Determines the Omega ratio of a strategy.
        See https://en.wikipedia.org/wiki/Omega_ratio for more details.
        """
        if periods ==1:
            return_threshold = required_return
        else:
            return_threshold = (1 + required_return) **(1. / periods) -1

        returns_less_thresh = returns - return_threshold
        numer = returns_less_thresh[returns_less_thresh >0.0].sum()
        denom = -1.0*returns_less_thresh[returns_less_thresh <0.0].sum()

        if denom >0.0:
            return numer / denom
        return np.nan

    def best(self,returns, aggregate=None, compounded=True):
        """Returns the best day/month/week/quarter/year's return"""

        return self.aggregate_returns(returns, aggregate, compounded).max()

    def worst(self,returns, aggregate=None, compounded=True):
        """Returns the worst day/month/week/quarter/year's return"""

        return self.aggregate_returns(returns, aggregate, compounded).min()
    def aggregate_returns(self,returns, period=None, compounded=True):
        """Aggregates returns based on date periods"""
        if period is None or 'day' in period:
            return returns
        index = returns.index

        if 'month' in period:
            return self.group_returns(returns, index.month, compounded=compounded)

        if 'quarter' in period:
            return self.group_returns(returns, index.quarter, compounded=compounded)

        if period == "A" or any(x in period for x in ['year', 'eoy', 'yoy']):
            return self.group_returns(returns, index.year, compounded=compounded)

        if 'week' in period:
            return self.group_returns(returns, index.week, compounded=compounded)

        if 'eow' in period or period == "W":
            return self.group_returns(returns, [index.year, index.week],
                                 compounded=compounded)

        if 'eom' in period or period == "M":
            return self.group_returns(returns, [index.year, index.month],
                                 compounded=compounded)

        if 'eoq' in period or period == "Q":
            return self.group_returns(returns, [index.year, index.quarter],
                                 compounded=compounded)

        if not isinstance(period, str):
            return self.group_returns(returns, period, compounded)

        return returns

    def group_returns(self,returns, groupby, compounded=False):
        """Summarize returns
        group_returns(df, df.index.year)
        group_returns(df, [df.index.year, df.index.month])
        """
        if compounded:
            return returns.groupby(groupby).apply(self.comp)
        return returns.groupby(groupby).sum()

    def compsum(self,returns):
        """Calculates rolling compounded returns"""
        return returns.add(1).cumprod() - 1

    def comp(self,returns):
        """Calculates total compounded returns"""
        return returns.add(1).prod() - 1

    def maximum_dd(self, returns,index_name):
        cm_return = (returns + 1).cumprod() - 1
        dd_price = 1e5 + 1e5 * (cm_return)

        # for MaximumDrawDown
        drowdown = (dd_price / np.maximum.accumulate(dd_price)) - 1.
        drowdown = drowdown.replace([np.inf, -np.inf, -0], 0)
        no_dd = drowdown == 0
        # extract dd start dates
        starts = ~no_dd & no_dd.shift(1)
        starts = list(starts[starts].index)

        # extract end dates
        ends = no_dd & (~no_dd).shift(1)
        ends = list(ends[ends].index)

        # no drawdown :)
        if not starts:
            return pd.DataFrame(
                index=[], columns=('start', 'valley', 'end', 'days',
                                   'max drawdown', '99% max drawdown'))

        # drawdown series begins in a drawdown
        if ends and starts[0] > ends[0]:
            starts.insert(0, drowdown.index[0])

        # series ends in a drawdown fill with last date
        if not ends or starts[-1] > ends[-1]:
            ends.append(drowdown.index[-1])

        # build dataframe from results
        data = []
        for i, _ in enumerate(starts):
            dd = drowdown[starts[i]:ends[i]]
            clean_dd = -self.remove_outliers(-dd, .99)
            data.append((starts[i], dd.idxmin(), ends[i],
                         (ends[i] - starts[i]).days,
                         dd.min() * 100, clean_dd.min() * 100))

        df = pd.DataFrame(data=data,
                          columns=('start', 'valley', 'end', 'days',
                                   'max drawdown',
                                   '99% max drawdown'))
        df['days'] = df['days'].astype(int)
        df['max drawdown'] = df['max drawdown'].astype(float)
        df['99% max drawdown'] = df['99% max drawdown'].astype(float)

        df['start'] = df['start'].dt.strftime('%Y-%m-%d')
        df['end'] = df['end'].dt.strftime('%Y-%m-%d')
        df['valley'] = df['valley'].dt.strftime('%Y-%m-%d')
        dd_stats = {
            f'{index_name}': {
                'Max Drawdown %': df.sort_values(
                    by='max drawdown', ascending=True
                )['max drawdown'].values[0] / 100,
                'Longest DD Days': str(np.round(df.sort_values(
                    by='days', ascending=False)['days'].values[0])),
                'Avg. Drawdown %': df['max drawdown'].mean() / 100,
                'Avg. Drawdown Days': str(np.round(df['days'].mean()))
            }
        }
        # pct multiplier
        pct = 100

        dd_stats = pd.DataFrame(dd_stats).T
        dd_stats['Max Drawdown %'] = dd_stats['Max Drawdown %'].astype(float) * pct
        dd_stats['Avg. Drawdown %'] = dd_stats['Avg. Drawdown %'].astype(float) * pct

        return dd_stats

    def remove_outliers(self, returns, quantile=.95):
        """Returns series of returns without the outliers"""
        return returns[returns < returns.quantile(quantile)]