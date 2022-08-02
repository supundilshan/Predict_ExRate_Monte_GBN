import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from pandas import read_csv

def Get_Trend(Orginal_data):
    L = len(Orginal_data)-1
    # i=0
    change = np.empty(L, dtype=float)
    trend = np.empty(L, dtype=int)

    for i in range(L):
        # print(i)
        change[i] = Orginal_data[i+1] - Orginal_data[i]

    for j in range(L):
        if change[j] > 0:
            trend[j] = 1
        else:
            trend[j] = 0
    # print(change)
    # print(len(change))
    # print(trend)
    # print(len(trend))

    return trend

def Initial_Probability(trenD):
    Total_Up_trends = trenD[np.where(trenD == 1)].size
    Total_Down_trends = trenD[np.where(trenD == 0)].size

    # Find initial probability
    Up_probability = Total_Up_trends/(Total_Up_trends+Total_Down_trends)
    Down_probability = Total_Down_trends / (Total_Up_trends + Total_Down_trends)
    Initial_Probability = [Up_probability, Down_probability]

    return Initial_Probability
    # print("initial Probability = ",Initial_Probability)

def Transition_Probability_Matrix(trenD):
    Total_Up_trends = trenD[np.where(trenD == 1)].size
    Total_Down_trends = trenD[np.where(trenD == 0)].size

    # Find Transition Probability Matrix
    if trenD[-1] == 1:
        Up_Trends = Total_Up_trends -1
        Down_Trends = Total_Down_trends
    else:
        Up_Trends = Total_Up_trends
        Down_Trends = Total_Down_trends - 1

    Movement = np.empty(len(trenD)-1, dtype=int)
    for k in range(len(trenD)-1):
        # up to up = 2
        if trenD[k+1] == trenD[k] & trenD[k+1] == 1:
            Movement[k] = 2
        # down to down = -2
        if trenD[k+1] == trenD[k] & trenD[k+1] == 0:
            Movement[k] = -2
        # down to up = 1
        if trenD[k+1] > trenD[k]:
            Movement[k] = 1
        # up to down = -1
        if trenD[k+1] < trenD[k]:
            Movement[k] = -1

    X11 = trenD[np.where(Movement == 2)].size / Up_Trends
    X12 = trenD[np.where(Movement == -1)].size / Up_Trends
    X21 = trenD[np.where(Movement == 1)].size / Down_Trends
    X22 = trenD[np.where(Movement == -2)].size / Down_Trends

    Matrix = [[X11 , X12],
              [X21 , X22]]
    return Matrix

def Find_Process_to_nth_Day(no_of_days,ini_prob, transition_matrix):
    Trend_up_to_nth_day = np.empty(no_of_days - 1, dtype=float)
    nth_matrix = transition_matrix

    for n in range(no_of_days-1):
        nth_matrix = np.dot(transition_matrix,nth_matrix)
        # print(nth_matrix)
        probability = np.dot(ini_prob,nth_matrix)
        # print(probability)
        # print('-------------------')
        Trend_up_to_nth_day[n] = probability[1]

        if probability[0] > probability[1]:
            Trend_up_to_nth_day[n] = 1
        else:
            Trend_up_to_nth_day[n] = 0

    return Trend_up_to_nth_day
    # print(Trend_up_to_nth_day)

def Marcow(Days):
    dataframe = read_csv("Data.csv")
    data = dataframe.values

    Trend = Get_Trend(data)
    # print(Trend)
    initial_probability = Initial_Probability(Trend)
    # print(initial_probability)

    transition_prob_matrix = Transition_Probability_Matrix(Trend)
    # print(transition_prob_matrix)

    trend_up_to_nth_day = Find_Process_to_nth_Day(Days,initial_probability,transition_prob_matrix)

    return trend_up_to_nth_day

if __name__ == "__main__":
    Trend_up_to_nth_Day =  Marcow(5)
    print(Trend_up_to_nth_Day)
    plt.plot(Trend_up_to_nth_Day, marker='.')