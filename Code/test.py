import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

print("START")

data = pd.read_csv(
    "Data/mystery_virus_daily_active_counts_RELEASE#1.csv"
)

plt.scatter(data['day'], data['active reported daily cases'])
plt.title("Active Cases Over Time")
plt.xlabel("Time (days)")
plt.ylabel("Active Reported Daily Cases")
plt.savefig("active_cases_plot.png")

print("END")