import numpy as np
from copulas.univariate import GaussianKDE
from copulas.bivariate import Clayton
from scipy.optimize import minimize
from scipy.stats import kendalltau, spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from text files
x_data = np.loadtxt('x_data.txt')
y_data = np.loadtxt('y_data.txt')

plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatterplot of Original Data')
plt.show()

# Marginal Density Plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.kdeplot(x_data, color='blue', label='X')
plt.title('Marginal Density Plot for X')
plt.subplot(1, 2, 2)
sns.kdeplot(y_data, color='orange', label='Y')
plt.title('Marginal Density Plot for Y')
plt.show()

# Correlation Heatmap
correlation_matrix = np.corrcoef(x_data, y_data)
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Marginal Distribution
marginals = [GaussianKDE(), GaussianKDE()]
marginals[0].fit(x_data)
marginals[1].fit(y_data)

# Transform data
transformed_x = marginals[0].cdf(x_data)
transformed_y = marginals[1].cdf(y_data)

# Define negative log-likelihood function for Clayton copula
def neg_log_likelihood(theta):
    copula = Clayton(theta)
    copula.fit(np.column_stack((transformed_x, transformed_y)))
    return -np.sum(np.log(copula.pdf(np.column_stack((transformed_x, transformed_y))))) 
    

# Initial guess for the parameter theta
initial_theta = 0.5  # Example initial value for theta

# Estimate Copula Parameter using MLE with different optimization methods

result_powell = minimize(neg_log_likelihood, initial_theta, method='Powell')
estimated_theta = result_powell.x[0]

print("Estimated theta using MLE:", estimated_theta)

# Copula
copula = Clayton(estimated_theta)
copula.fit(np.column_stack((transformed_x, transformed_y)))

# Generate Synthetic Data using Copula
synthetic_data = copula.sample(100)

np.savetxt('synthetic_data.csv', synthetic_data, delimiter=',')

# Copula Density Plot
plt.figure(figsize=(8, 6))
sns.kdeplot(synthetic_data[:, 0], synthetic_data[:, 1], cmap="Blues", shade=True, shade_lowest=False)
plt.xlabel('Transformed X')
plt.ylabel('Transformed Y')
plt.title('Copula Density Plot')
plt.show()

# QQ Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(np.sort(synthetic_data[:, 0]), np.sort(transformed_x), color='blue', label='X')
sns.scatterplot(np.sort(synthetic_data[:, 1]), np.sort(transformed_y), color='orange', label='Y')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.title('QQ Plot')
plt.legend()
plt.show()

# Calculate Kendall's Tau
kendall_tau, _ = kendalltau(synthetic_data[:, 0], synthetic_data[:, 1])

# Calculate Spearman's Rank Correlation
spearman_corr, _ = spearmanr(synthetic_data[:, 0], synthetic_data[:, 1])

# Calculate Pearson Correlation Coefficient
pearson_corr, _ = pearsonr(synthetic_data[:, 0], synthetic_data[:, 1])

# Print the calculated measures
print("Kendall's Tau:", kendall_tau)
print("Spearman's Rank Correlation:", spearman_corr)
print("Pearson Correlation Coefficient:", pearson_corr)

# Calculate upper tail dependence coefficient
quantile_level = 0.9
upper_tail_dependence = np.mean((synthetic_data[:, 0] > np.quantile(synthetic_data[:, 0], quantile_level)) & 
                                (synthetic_data[:, 1] > np.quantile(synthetic_data[:, 1], quantile_level)))

# Determine if copula exhibits upper tail dependence
print("Upper tail dependence coefficient", upper_tail_dependence)
if upper_tail_dependence > 0:
    print("Copula exhibits upper tail dependence")
else:
    print("Copula does not exhibit upper tail dependence")
    
quantile_level = 0.1
lower_tail_dependence = np.mean((synthetic_data[:, 0] < np.quantile(synthetic_data[:, 0], quantile_level)) & 
                                (synthetic_data[:, 1] < np.quantile(synthetic_data[:, 1], quantile_level)))

# Determine if copula exhibits lower tail dependence
print("Lower tail dependence coefficient", lower_tail_dependence)
if lower_tail_dependence > 0:
    print("Copula exhibits lower tail dependence")
else:
    print("Copula does not exhibit lower tail dependence")
    
# Dependence Structure Plots (Kendall's Tau, Spearman's Rank Correlation, Pearson Correlation Coefficient)
quantile_levels = np.linspace(0.1, 0.9, 9)
kendall_tau = []
spearman_corr = []
pearson_corr = []

for level in quantile_levels:
    kendall_tau.append(kendalltau(synthetic_data[:, 0] > np.quantile(synthetic_data[:, 0], level),
                                  synthetic_data[:, 1] > np.quantile(synthetic_data[:, 1], level))[0])
    spearman_corr.append(spearmanr(synthetic_data[:, 0] > np.quantile(synthetic_data[:, 0], level),
                                    synthetic_data[:, 1] > np.quantile(synthetic_data[:, 1], level))[0])

plt.figure(figsize=(10, 6))
plt.plot(quantile_levels, spearman_corr, label="Spearman's Rank Correlation")
plt.xlabel('Quantile Level')
plt.ylabel('Dependence Measure')
plt.title('Dependence Structure')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(quantile_levels, kendall_tau, label="Kendall's Tau")
plt.xlabel('Quantile Level')
plt.ylabel('Dependence Measure')
plt.title('Dependence Structure')
plt.legend()
plt.show()
# Tail Dependence Plots
upper_tail_dependence = []
lower_tail_dependence = []

for level in quantile_levels:
    upper_tail_dependence.append(np.mean((synthetic_data[:, 0] > np.quantile(synthetic_data[:, 0], level)) & 
                                         (synthetic_data[:, 1] > np.quantile(synthetic_data[:, 1], level))))
    lower_tail_dependence.append(np.mean((synthetic_data[:, 0] < np.quantile(synthetic_data[:, 0], 1 - level)) & 
                                         (synthetic_data[:, 1] < np.quantile(synthetic_data[:, 1], 1 - level))))

plt.figure(figsize=(10, 6))
plt.plot(quantile_levels, upper_tail_dependence, label="Upper Tail Dependence")
plt.plot(quantile_levels, lower_tail_dependence, label="Lower Tail Dependence")
plt.xlabel('Quantile Level')
plt.ylabel('Tail Dependence Coefficient')
plt.title('Tail Dependence')
plt.legend()
plt.show()

num_simulations = 100
synthetic_datasets = [copula.sample(100) for _ in range(num_simulations)]
   
plt.figure(figsize=(8, 6))
sns.histplot(synthetic_datasets[0][:, 0], kde=True, color='blue', label='Synthetic X')
sns.histplot(synthetic_datasets[0][:, 1], kde=True, color='orange', label='Synthetic Y')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Synthetic Data')
plt.legend()
plt.show()    
    
# Sensitivity Analysis of Copula Parameters (for example, theta)
initial_guesses = np.linspace(0.1, 3.0, 20)
estimated_thetas = []

for initial_theta in initial_guesses:
    result = minimize(neg_log_likelihood, initial_theta, method='Powell')
    estimated_theta = result.x[0]
    estimated_thetas.append(estimated_theta)

plt.figure(figsize=(8, 6))
plt.plot(initial_guesses, estimated_thetas, marker='o', linestyle='-')
plt.xlabel('Initial Guess of Theta')
plt.ylabel('Estimated Theta')
plt.title('Sensitivity Analysis of Initial Guess vs Estimated Theta')
plt.grid(True)
plt.show()

# Empirical Copula vs. Theoretical Copula
plt.figure(figsize=(8, 6))
plt.scatter(transformed_x, transformed_y, alpha=0.5, label='Empirical Copula')
sns.kdeplot(synthetic_data[:, 0], synthetic_data[:, 1], cmap="Blues", shade=True, shade_lowest=False, label='Theoretical Copula')
plt.xlabel('Transformed X')
plt.ylabel('Transformed Y')
plt.title('Empirical vs. Theoretical Copula')
plt.legend()
plt.show()   
    
