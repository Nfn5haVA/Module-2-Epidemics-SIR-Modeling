#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math

#%%
# Load the data
data = pd.read_csv('C:\\Users\karin\OneDrive - University of Virginia\\Second Year\\Comp BME\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)
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

plt.show()