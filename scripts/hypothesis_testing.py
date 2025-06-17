import pandas as pd
from scipy import stats
import numpy as np

# Load data with low_memory=False to avoid DtypeWarning
df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)

# Calculate LossRatio, handling zero/negative TotalPremium
df['LossRatio'] = np.where(df['TotalPremium'] > 0, df['TotalClaims'] / df['TotalPremium'], np.nan)

# Clean data: Remove NaN, inf, and outliers
df = df.dropna(subset=['LossRatio', 'TotalClaims', 'Province', 'VehicleType'])
df = df[np.isfinite(df['LossRatio'])]

# Hypothesis 1: Loss ratio in Gauteng vs Western Cape
gauteng = df[df['Province'] == 'Gauteng']['LossRatio']
western_cape = df[df['Province'] == 'Western Cape']['LossRatio']
if len(gauteng) > 1 and len(western_cape) > 1:
    t_stat, p_value = stats.ttest_ind(gauteng, western_cape, equal_var=False)
    print(f"H1: Gauteng vs Western Cape Loss Ratio")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
else:
    print("H1: Insufficient data for Gauteng or Western Cape")

# Hypothesis 2: TotalClaims by Vehicle Type
passenger = df[df['VehicleType'] == 'Passenger Vehicle']['TotalClaims']
heavy = df[df['VehicleType'] == 'Heavy Commercial']['TotalClaims']
if len(passenger) > 1 and len(heavy) > 1:
    t_stat, p_value = stats.ttest_ind(passenger, heavy, equal_var=False)
    print(f"H2: Passenger vs Heavy Commercial TotalClaims")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
else:
    print("H2: Insufficient data for Passenger or Heavy Commercial")

# Hypothesis 3: TotalPremium by Gender
male = df[df['Gender'] == 'Male']['TotalPremium']
female = df[df['Gender'] == 'Female']['TotalPremium']
if len(male) > 1 and len(female) > 1:
    t_stat, p_value = stats.ttest_ind(male, female, equal_var=False)
    print(f"H3: Male vs Female TotalPremium")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
else:
    print("H3: Insufficient data for Male or Female")

# Save results
with open('scripts/hypothesis_results.txt', 'w') as f:
    f.write("Hypothesis Testing Results\n")
    f.write(f"H1: Gauteng vs Western Cape Loss Ratio, P-value = {p_value if 'p_value' in locals() else 'N/A'}\n")
    f.write(f"H2: Passenger vs Heavy Commercial TotalClaims, P-value = {p_value if 'p_value' in locals() else 'N/A'}\n")
    f.write(f"H3: Male vs Female TotalPremium, P-value = {p_value if 'p_value' in locals() else 'N/A'}\n")