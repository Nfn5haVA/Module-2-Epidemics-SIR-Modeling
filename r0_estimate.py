import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math


# Load the data
data = pd.read_csv(
    "../Data/mystery_virus_daily_active_counts_RELEASE#1.csv",
    parse_dates=["date"]
)

# Extract time  and active cases
days = data["day"].to_numpy(dtype=float)
active_cases = data["active reported daily cases"].to_numpy(dtype=float)

# Remove zero values so the graph can be plotted 
mask = active_cases > 0
days = days[mask]
active_cases = active_cases[mask]


# Exponential model: I(t) = A * exp(r t)
def exponential_growth(t, A, r):
    return A * np.exp(r * t)

# Initial parameters
initial_guess = (active_cases[0], 0.1)

# Fit the model
params, covariance = curve_fit(exponential_growth, days, active_cases, p0=initial_guess)
A_hat, r_hat = params

# 
# Estimate R0 and  Infectious period range
D_low = 7.0
D_high = 11.0

R0_low = math.exp(r_hat * D_low)
R0_high = math.exp(r_hat * D_high)

print(f"Estimated growth rate r = {r_hat:.5f} per day")
print(f"Estimated R0 range (D=7–11 days): {R0_low:.3f} – {R0_high:.3f}")


# Plot data + exponential fit

plt.figure(figsize=(8,5))
plt.scatter(days, active_cases, label="Active cases ")

t_fit = np.linspace(days.min(), days.max(), 400)
I_fit = exponential_growth(t_fit, A_hat, r_hat)

plt.plot(t_fit, I_fit, color="red", label="Exponential fit")

plt.title("Exponential Growth of Active Cases")
plt.xlabel("Time in days")
plt.ylabel("Active Reported Daily Cases")
plt.legend()
plt.tight_layout()
plt.show()