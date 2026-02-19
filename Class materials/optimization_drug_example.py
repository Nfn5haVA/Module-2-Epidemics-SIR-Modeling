# drug efficacy optimization example for BME 2315
# made by Lavie, fall 2025

#%% import libraries
import numpy as np
import matplotlib.pyplot as plt


#%% define drug models

# define toxicity levels for each drug (lambda)
metformin_lambda = 0.5

lisinopril_lambda = 0.8

escitalopram_lambda = 0.3

def metformin(x):   # mild toxicity, moderate efficacy
    efficacy = 0.8 * np.exp(-0.1*(x-5)**2)
    toxicity = 0.2 * x**2 / 100
    return efficacy - metformin_lambda * toxicity
def lisinopril(x):  # strong efficacy, higher toxicity
    efficacy = np.exp(-0.1*(x-7)**2)
    toxicity = 0.3 * x**2 / 80
    return efficacy - lisinopril_lambda * toxicity
def escitalopram(x):  # weaker efficacy, low toxicity
    efficacy = 0.6 * np.exp(-0.1*(x-4)**2)
    toxicity = 0.1 * x**2 / 120
    return efficacy - escitalopram_lambda * toxicity

def total(x): 
    return metformin(x) + lisinopril(x) + escitalopram(x)




#%% plot drug efficacies
x = np.linspace(0, 15, 100)
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(x, metformin(x), label='Metformin', color='blue')
plt.plot(x, lisinopril(x), label='Lisinopril', color='orange')
plt.plot(x, escitalopram(x), label='Escitalopram', color='green')
plt.plot(x, metformin(x) + lisinopril(x) + escitalopram(x), label='Combined Effect', color='red', linestyle='--')
plt.title('Drug Efficacy vs Dosage')
plt.xlabel('Dosage (mg)')
plt.ylabel('Net Effect')
plt.legend()
plt.show()

# %% Find optimal dosages for each drug

# First method: Steepest Ascent using the update rule

# first, need the first derivative (gradient)
def gradient(f, x, h=1e-4):
    """Central difference approximation for f'(x)."""
    return (f(x + h) - f(x - h)) / (2*h)

def steepest_ascent(f, x0, h_step=0.1, tol=1e-6, max_iter=1000):
    x = x0 # update initial guess
    for i in range(max_iter):
        grad = gradient(f, x)
        x_new = x + h_step * grad     
        
        if abs(x_new - x) < tol:      # convergence condition, when solution is 0
            print(f"Converged in {i+1} iterations.")
            break
            
        x = x_new
    return x, f(x)

# metformin
opt_dose_metformin, opt_effect_metformin = steepest_ascent(metformin, x0=1.0)
print(f"Steepest Ascent Method - Optimal Metformin Dose: {opt_dose_metformin:.2f} mg")
print(f"Steepest Ascent Method - Optimal Metformin Effect: {opt_effect_metformin*100:.2f}%")

# lisinopril
opt_dose_lisinopril, opt_effect_lisinopril = steepest_ascent(lisinopril, x0=1.0)
print(f"Steepest Ascent Method - Optimal Lisinopril Dose: {opt_dose_lisinopril:.2f} mg")
print(f"Steepest Ascent Method - Optimal Lisinopril Effect: {opt_effect_lisinopril*100:.2f}%")

# escitalopram
opt_dose_escitalopram, opt_effect_escitalopram = steepest_ascent(escitalopram, x0=1.0)
print(f"Steepest Ascent Method - Optimal Escitalopram Dose: {opt_dose_escitalopram:.2f} mg")
print(f"Steepest Ascent Method - Optimal Escitalopram Effect: {opt_effect_escitalopram*100:.2f}%")

opt_dose_total, opt_effect_total = steepest_ascent(total, x0=1.0)
print(f"Steepest Ascent Method - Optimal Total Dose: {opt_dose_total:.2f} mg")
print(f"Steepest Ascent Method - Optimal Total Effect: {opt_effect_total*100:.2f}%")

# %% Newton's method

# requires second derivative
def second_derivative(f, x, h=1e-4):
    """Central difference approximation for f''(x)."""
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

def newtons_method(f, x0, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        grad = gradient(f, x)
        hess = second_derivative(f, x)
        
        if hess == 0:  # avoid division by zero
            print("Zero second derivative. No solution found.")
            return x, f(x)
        
        x_new = x - grad / hess
        
        if abs(x_new - x) < tol:
            print(f"Converged in {i+1} iterations.")
            break
            
        x = x_new
    return x, f(x)

# metformin
opt_dose_metformin_nm, opt_effect_metformin_nm = newtons_method(metformin, x0=1.0)
print(f"Newton's Method - Optimal Metformin Dose: {opt_dose_metformin_nm:.2f} mg")
print(f"Newton's Method - Optimal Metformin Effect: {opt_effect_metformin_nm*100:.2f}%")                

# lisinopril
opt_dose_lisinopril_nm, opt_effect_lisinopril_nm = newtons_method(lisinopril, x0=1.0)
print(f"Newton's Method - Optimal Lisinopril Dose: {opt_dose_lisinopril_nm:.2f} mg")
print(f"Newton's Method - Optimal Lisinopril Effect: {opt_effect_lisinopril_nm*100:.2f}%")

# escitalopram
opt_dose_escitalopram_nm, opt_effect_escitalopram_nm = newtons_method(escitalopram, x0=1.0)
print(f"Newton's Method - Optimal Escitalopram Dose: {opt_dose_escitalopram_nm:.2f} mg")
print(f"Newton's Method - Optimal Escitalopram Effect: {opt_effect_escitalopram_nm*100:.2f}%")

opt_dose_total, opt_effect_total = steepest_ascent(total, x0=1.0)
print(f"Steepest Ascent Method - Optimal Total Dose: {opt_dose_total:.2f} mg")
print(f"Steepest Ascent Method - Optimal Total Effect: {opt_effect_total*100:.2f}%")



def metformin_new(x, new_lambda):   # mild toxicity, moderate efficacy
    efficacy = 0.8 * np.exp(-0.1*(x-5)**2)
    toxicity = 0.2 * x**2 / 100
    return efficacy - new_lambda * toxicity

opt_dose_new, opt_effect_new = newtons_method(total, x0=1.0)
print(f"Newton's Method with New Lambda - Optimal Total Dose: {opt_dose_new:.2f} mg")
print(f"Newton's Method with New Lambda - Optimal Total Effect: {opt_effect_new*100:.2f}%")


new_lambda = 0
new_dose = 0
current_best = 10000
for number in np.arange(0.05, 5, 0.01): # this line was written with the help of Copilot
    def metformin_func(x, lam = number):
        return metformin_new(x, lam)

    opt_dose, opt_effect = newtons_method(metformin_func, x0 = 1.0)
    comparison = abs(opt_dose_total - opt_dose) #copilot suggested using abs() for this line

    if comparison < current_best:
        current_best = comparison
        new_lambda = number
        new_dose = opt_dose

print(f"Best Lambda: {new_lambda} | Optimal Dose: {new_dose}")

    