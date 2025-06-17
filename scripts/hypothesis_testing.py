import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Load data
df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|')
df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']

# Hypothesis 1: Loss ratio in Gauteng vs Western Cape
gauteng = df[df['Province'] == 'Gauteng']['LossRatio'].dropna()
western_cape = df[df['Province'] == 'Western Cape']['LossRatio'].dropna()
t_stat, p_value = stats.ttest_ind(gauteng, western_cape, equal_var=False)
print(f"H1: Gauteng vs Western Cape Loss Ratio")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

# Hypothesis 2: TotalClaims by Vehicle Type
passenger = df[df['VehicleType'] == 'Passenger Vehicle']['TotalClaims'].dropna()
heavy = df[df['VehicleType'] == 'Heavy Commercial']['TotalClaims'].dropna()
t_stat, p_value = stats.ttest_ind(passenger, heavy, equal_var=False)
print(f"H2: Passenger vs Heavy Commercial TotalClaims")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

# Save results
with open('scripts/hypothesis_results.txt', 'w') as f:
    f.write("Hypothesis Testing Results\n")
    f.write(f"H1: P-value = {p_value:.4f}\n")
    f.write(f"H2: P-value = {p_value:.4f}\n")