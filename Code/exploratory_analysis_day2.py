#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math

#%%
# Load the data
data = pd.read_csv("C:\\Users\\karin\\OneDrive - University of Virginia\\Second Year\\Comp BME\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#2.csv", parse_dates=['date'], header=0, index_col=None)
#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)

days = []
active_cases = []

for day in data['day']:
    days.append(day)

for case in data['active reported daily cases']:
    active_cases.append(case)

# Fit the exponential growth model to the data. 
# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.

opt_r, covariance = curve_fit(exponential_growth, days, active_cases)
# Approximate R0 using this fit
ip_low = 7 #we know that the infectious period is 7-11 days, so we will use those to calculate a range for R0
ip_high = 11

rdprod_low = ip_low * opt_r # finding rD for low and high ends of infectious period
rdprod_high = ip_high * opt_r

low_r0 = math.e ** rdprod_low
high_r0 = math.e ** rdprod_high

print(f"r0 Range: {low_r0} - {high_r0}") 


# Add the fit as a line on top of your scatterplot.

plt.scatter(days, active_cases)
plt.title("Active Cases Over Time")
plt.xlabel("Time (days)")
plt.ylabel("Active Reported Daily Cases")

t_axis = np.linspace(1, max(days), 400) # creates a an array of 400 evenly spaced points between one and the number of days in the csv file
y_axis = exponential_growth(t_axis,opt_r[0])
plt.plot(t_axis, y_axis, color = 'red') #plot a line of the fit on top of the scatterplot

#plt.show()

#1/infectious period, which we know is 7-11 days
gamma_high = 0.143
gamma_low = 0.091 

# 1/incubation period, which we know is 12-18 days
sigma_high = 0.0833
sigma_low = 0.0556
# estimated R0 range * gamma range 
beta_low = 0.211
beta_high = 0.534

sse = 1000
N = 17612
i0 = 1
r0 = 0
e0 = 2
s0 = N - e0 - i0 - r0
def eulers(beta, sigma, gamma, s0, e0, i0, r0, timepoints, N):
    s = [s0]
    e = [e0]
    i = [i0]
    r = [r0]

    for timepoint in range(len(timepoints)):
        dS_dt = -1 * beta * s[timepoint] * i[timepoint] / N # calculate the derivative at the current timepoint based on the equations from class
        s.append(s[timepoint] + dS_dt)

        dE_dt = (beta * s[timepoint] * i[timepoint] / N) - (sigma * e[timepoint])
        e.append(e[timepoint] + dE_dt)

        dI_dt = (sigma * e[timepoint]) - (gamma * i[timepoint])
        i.append(i[timepoint] + dI_dt)

        dR_dt = gamma * i[timepoint]
        r.append(r[timepoint] + dR_dt) 
    
    return s, e, i, r


def optimization():       
    sse = 10000000000000000
    best_b = 0
    best_s = 0
    best_g = 0

    # iterate through every combination of beta, gamma, and sigma, calculating sse for each combo and determining the best parameters 
    for beta in np.arange(beta_low, beta_high, 0.01):
        for sigma in np.arange(sigma_low, sigma_high, 0.01):
            for gamma in np.arange(gamma_low, gamma_high, 0.01):
                s, e, i, r = eulers(beta, sigma, gamma, s0, e0, i0, r0, days, N)
                sse_new = np.sum((np.array(i[:len(data)]) - active_cases)**2)
                #print("placeholder")
                #sse_new = np.sum((i - y_axis)**2)
                if sse_new < sse:
                    sse = sse_new
                    best_b = beta
                    best_s = sigma
                    best_g = gamma
    
    return best_b, best_s, best_g, sse


best_beta, best_ssigma, best_gamma, sse= optimization()




def prediction(best_beta, best_sigma, best_gamma):
    # set up the time points in the furture to run the predicted model. 
    future_days = list(range(500))
    # Run Euler's method far into the future to find the peak
    s, e, i, r = eulers(best_beta, best_sigma, best_gamma, s0, e0, i0, r0, future_days, N)
    # Find the peak number of infected individuals and the day it occurs. sitting i to numbers of days 
    peak_value = max(i)
    peak_day = i.index(peak_value)
    
    print(f"Best beta: {best_beta:.4f}, Best sigma: {best_sigma:.4f}, Best gamma: {best_gamma:.4f}")
    print(f"Peak infections: {peak_value:.0f} people")
    print(f"Peak occurs on day: {peak_day}")
    # Calculate the percentage of the population that is infected at the peak
    print(f"That is {peak_value/N*100:.1f}% of the population ({N} total)")
    
    # Plot the SEIR model's infected curve over time
    # use the infected compartment curve over the 500 predicted days
    plt.figure()
    plt.plot(i[:len(future_days)], label='SEIR Model (Infected)', color='blue')
    plt.axvline(x=peak_day, color='red', linestyle='--', label=f'Peak Day: {peak_day}')
    plt.axhline(y=peak_value, color='orange', linestyle='--', label=f'Peak Value: {peak_value:.0f}')
    plt.title("SEIR Model - Predicted Peak")
    plt.xlabel("Day")
    plt.ylabel("Infected")
    plt.legend()
    plt.show()
    
    return peak_value, peak_day

# Call the function
peak_value, peak_day = prediction(best_beta, best_ssigma, best_gamma)
