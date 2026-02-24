#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv('C:\\Users\\karin\\OneDrive - University of Virginia\\Second Year\\Comp BME\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

#%%
# Make a plot of the active cases over time

days = []
active_cases = []

for day in data['day']:
    days.append(day)

for case in data['active reported daily cases']:
    active_cases.append(case)


plt.scatter(days, active_cases)
plt.title("Active Cases Over Time")
plt.xlabel("Time (days)")
plt.ylabel("Active Reported Daily Cases")
plt.show()