import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from pandas import read_csv

def Montecarlo(Days,Trials):

    data = read_csv("Data.csv")

    # dataframe = read_csv("Data.csv")
    # dataset = dataframe.values
    # dataset1 = dataset[:, 1]
    # data = dataset1.astype(np.float)

    # pct=data.pct_change()
    # pct1=1+pct
    log_returns = np.log(1 + data.pct_change())
    # log return=ln(ccurrent price)-ln(prev price)
    #Plot
    # sns.distplot(log_returns.iloc[1:])
    # plt.xlabel("Daily Return")
    # plt.ylabel("Frequency")

    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5*var)

    stdev = log_returns.std()
    days = Days
    trials = Trials
    Z = norm.ppf(np.random.rand(days, trials)) #days, trials
    daily_returns = np.exp(drift.values + stdev.values * Z)

    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = data.iloc[-1]
    for t in range(1, days):
        price_paths[t] = price_paths[t-1]*daily_returns[t]

    # plot all out comes
    # plt.plot(price_paths)

    return price_paths

if __name__ == "__main__":
    PricePaths = Montecarlo(5,5)
    # plot all out comes
    plt.plot(PricePaths)