#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math

#%%
# Load the data
data = pd.read_csv("C:\\Users\\karin\\OneDrive - University of Virginia\\Second Year\\Comp BME\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#3.csv", parse_dates=['date'], header=0, index_col=None)
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

#t_axis = np.linspace(1, max(days), 400) # creates a an array of 400 evenly spaced points between one and the number of days in the csv file
#y_axis = exponential_growth(t_axis,opt_r[0])
#plt.plot(t_axis, y_axis, color = 'red') #plot a line of the fit on top of the scatterplot

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


def optimization(b_low, b_high, s_low, s_high, g_low, g_high, start_day, s_0, e_0, i_0, r_0):       
    sse = 10000000000000000
    best_b = 0
    best_s = 0
    best_g = 0

    # iterate through every combination of beta, gamma, and sigma, calculating sse for each combo and determining the best parameters 
    for beta in np.arange(b_low, b_high, 0.01):
        for sigma in np.arange(s_low, s_high, 0.01):
            for gamma in np.arange(g_low, g_high, 0.01):
                s, e, i, r = eulers(beta, sigma, gamma, s_0, e_0, i_0, r_0, days, N)
                sse_new = np.sum((np.array(i[:len(data)]) - active_cases)**2)
                #print("placeholder")
                #sse_new = np.sum((i - y_axis)**2)
                if sse_new < sse:
                    sse = sse_new
                    best_b = beta
                    best_s = sigma
                    best_g = gamma
    
    return best_b, best_s, best_g, sse

best_beta, best_sigma, best_gamma, sse= optimization(beta_low, beta_high, sigma_low, sigma_high, gamma_low, gamma_high, 0, s0, e0, i0, r0)
s, e, i, r = eulers(best_beta, best_sigma, best_gamma, s0, e0, i0, r0, days, N)

print(f" lol {best_beta}, {best_sigma}, {best_gamma}")



def prediction(best_beta, best_sigma, best_gamma, start_day, end_day):
    # set up the time points in the furture to run the predicted model. 
    future_days = list(range(start_day, end_day))
    # Run Euler's method far into the future to find the peak
    susceptible, exposed, infected, recovered = eulers(best_beta, best_sigma, best_gamma, s[start_day], e[start_day], i[start_day], r[start_day], future_days, N)
    # Find the peak number of infected individuals and the day it occurs. sitting i to numbers of days 
    peak_value = max(i)
    peak_day = i.index(peak_value) + start_day
    
    print(f"Best beta: {best_beta:.4f}, Best sigma: {best_sigma:.4f}, Best gamma: {best_gamma:.4f}")
    print(f"Peak infections: {peak_value:.0f} people")
    print(f"Peak occurs on day: {peak_day}")
    # Calculate the percentage of the population that is infected at the peak
    print(f"That is {peak_value/N*100:.1f}% of the population ({N} total)")

    return peak_value, peak_day, susceptible, exposed, infected, recovered, future_days

def graph(future_days, i, color, start_day, label):
    peak_value = max(i)
    peak_day = i.index(peak_value) + start_day
    plt.plot(future_days, i[:len(future_days)], label= label, color=color)
    #plt.axvline(x=peak_day, color='red', linestyle='--', label=f'Peak Day: {peak_day}')
    #plt.axhline(y=peak_value, color='orange', linestyle='--', label=f'Peak Value: {peak_value:.0f}')



# Call the function
best_beta, best_sigma, best_gamma, sse= optimization(beta_low, beta_high, sigma_low, sigma_high, gamma_low, gamma_high, 0, s0, e0, i0, r0)
peak_value, peak_day, s, e, i, r, future_days  = prediction(best_beta, best_sigma, best_gamma, 0, 70)

#graph(future_days, i, 'blue', 0)



# error prediction:

def calc_error():
    highest_day = 0
    highest_infection = 0
    for i in range(len(active_cases)):
        if active_cases[i] > highest_infection:
            highest_day = i 
            highest_infection = active_cases[i]
        
    #error in peak number of cases
    true_peak_num = highest_infection 
    pred_peak_num = peak_value 
    peak_num_error = ((true_peak_num - pred_peak_num) / true_peak_num) * 100

    # error in peak days
    true_peak_day = highest_day 
    pred_peak_day = peak_day
    peak_day_error = ((true_peak_day - pred_peak_day) / true_peak_day) * 100

    return f"True Percent Error in the Peak Number of Cases: {peak_num_error}%  \n   True Percent Error in Peak Days: {peak_day_error}"

#print(calc_error())

def testing_model():
    new_ip_high = 9
    new_ip_low = 5
    new_gh = 1/new_ip_high
    new_gl = 1/new_ip_low
    second_b, second_s, second_g, _ = optimization(beta_low, beta_high, sigma_low, sigma_high, new_gh, new_gl, 69, s[69], e[69], i[69], r[69])
    second_pv, second_pd, s2, e2, i2, r2, future_days2 = prediction(second_b, second_s, second_g, 70, 120)
    print(i2)
    graph(future_days2, i2, "red", 70, 'Testing and Quarantine')


def vaccine_model():
    b_val, s_val, g_val, _ = optimization(beta_low, beta_high, sigma_low, sigma_high, gamma_low, gamma_high, 0, s0, e0, i0, r0)
    pday, value, vs, ve, vi, vr, future_daysv = prediction(b_val, s_val, g_val, 0, 70)

    print(F"s: {vs[70]}, e: {ve[70]}, i: {vi[70]}, r: {vr[70]}")

    new_s0 = vs[70] - (0.9 *2000)
    new_e0 = ve[70]
    new_i0 = vi[70]
    new_vr = vr[70] + (0.9 * 2000)

    b2_val, s2_val, g2_val, _ = optimization(beta_low, beta_high, sigma_low, sigma_high, gamma_low, gamma_high, 70, new_s0, new_e0, new_i0, new_vr)
    pday2, value2, vs2, ve2, vi2, vr2, future_daysv2 = prediction(b2_val, s2_val, g2_val, 70, 120)
    graph(future_daysv2,vi2, "purple", 70, 'Single Event Vaccine') 


first_b, first_s, first_g, _1 = optimization(beta_low, beta_high, sigma_low, sigma_high, gamma_low, gamma_high, 0, s0, e0, i0, r0)
first_pv, first_pd, s1, e1, i1, r1, future_days1 = prediction(first_b, first_s, first_g, 0, 120)
graph(future_days1, i1, "blue", 0, 'SEIR Model (Infected)')
testing_model()
vaccine_model()
plt.title("SEIR Model - Predicted Peak")
plt.xlabel("Day")
plt.ylabel("Infected")
plt.legend()
plt.show()
    




