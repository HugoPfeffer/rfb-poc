# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats.mstats import winsorize
import scipy.stats as stats
import statsmodels.api as sm

# %%
# Parameters
mean_return = 0.24
std_dev = 0.10  # Adjust for reasonable variability (10% annual volatility)

# Generate synthetic returns
synthetic_data = np.random.normal(loc=mean_return, scale=std_dev, size=1000)

# %%
# Winsorize data (limit extreme outliers to the 1st and 99th percentiles)
cleaned_data = winsorize(synthetic_data, limits=[0.01, 0.01])

# %%
# Adding varying volatility
crisis_periods = np.random.choice([0.10, 0.30], size=1000, p=[0.9, 0.1])
synthetic_data = np.random.normal(loc=mean_return, scale=crisis_periods)

# %%
print(synthetic_data)

# %%
sns.histplot(cleaned_data, kde=True)
plt.title("Synthetic Annual Returns")
plt.show()

# %%
# Summary statistics
mean = np.mean(synthetic_data)
std = np.std(synthetic_data)
skewness = stats.skew(synthetic_data)
kurtosis = stats.kurtosis(synthetic_data)

print(f"Mean: {mean:.4f}")
print(f"Standard Deviation: {std:.4f}")
print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurtosis:.4f}")

# %%
sns.histplot(synthetic_data, kde=True)
plt.title("Histogram and KDE Plot of Synthetic Returns")
plt.xlabel("Annual Return")
plt.ylabel("Frequency")
plt.show()

# %%
sns.boxplot(x=synthetic_data)
plt.title("Boxplot of Synthetic Returns")
plt.xlabel("Annual Return")
plt.show()

# %%

sm.qqplot(synthetic_data, line='s')
plt.title("QQ Plot of Synthetic Returns")
plt.show()


