import matplotlib.pyplot as plt
import numpy as np
import statistics

import Marcow as mr
import MonteCarlo as monte

if __name__ == '__main__':
    Days = 5
    Simulations = 100

    Markow_out = mr.Marcow(Days)
    # print(Markow_out)

    figure, axis = plt.subplots(2)
    PricePaths = monte.Montecarlo(Days, Simulations)
    all_mean = np.mean(PricePaths[-1])

    SUM_of_squired_error = np.sum(np.square(202.96155 - PricePaths[-1]))
    Calculated_MSE = SUM_of_squired_error / 100
    min(PricePaths[-1])
    statistics.median(PricePaths[-1])
    max(PricePaths[-1])
    print("Calculated_MSE : ", Calculated_MSE)


    axis[0].plot(PricePaths)
    axis[0].set_title("Results of Montecarlo Simulation")
    # axis[0].set_title("Results of Montecarlo Simulation Mean = %0.2f" % all_mean)

    Reduced_Sum = 0
    sum_of_sq_error = 0
    n=0
    index = np.empty(100, dtype=int)
    for i in range(Simulations):
        x = PricePaths[:,i]
        array = mr.Get_Trend(x)
        test = np.array_equal(Markow_out, array)
        if test:
            Reduced_Sum = Reduced_Sum + x[-1]
            sum_of_sq_error = sum_of_sq_error + np.square(202.96155 - x[-1])
            n = n+1
            axis[1].plot(PricePaths[:,i],marker='.')

    print("sum_of_sq_error",sum_of_sq_error)
    Resuced_Mean = Reduced_Sum /n
    # axis[1].set_title("Reduced Results from Montecarlo Simulation Mean = %0.2f" % Resuced_Mean)
    axis[1].set_title("Reduced Results from Montecarlo Simulation")