import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set Seaborn style
sns.set_style('whitegrid')
sns.set_palette('muted')

# Create data directory
if not os.path.exists('data'):
    os.makedirs('data')

# Load data
try:
    df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|')
except FileNotFoundError:
    print("Dataset 'MachineLearningRating_v3.txt' not found in 'data/' folder.")
    exit()
except pd.errors.ParserError:
    print("Error parsing 'MachineLearningRating_v3.txt'. Trying tab separator...")
    try:
        df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='\t')
    except:
        print("Tab separator failed. Please check the file format and separator.")
        exit()

# Normalize column names (strip whitespace, lowercase)
df.columns = df.columns.str.strip().str.lower()

# Verify columns
required_cols = ['totalpremium', 'totalclaims', 'customvalueestimate', 'province', 'vehicletype']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}. Available columns: {list(df.columns)}")
    exit()

# 1. Data Summarization
print("=== Descriptive Statistics ===")
numerical_cols = ['totalpremium', 'totalclaims', 'customvalueestimate']
print(df[numerical_cols].describe())

print("\n=== Data Types ===")
print(df.dtypes)

# 2. Data Quality Assessment
print("\n=== Missing Values ===")
missing = df.isnull().sum()
print(missing[missing > 0])

# 3. Univariate Analysis
try:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col].dropna(), ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('scripts/histograms.png')
    plt.close()
except Exception as e:
    print(f"Error in univariate histograms: {e}")

try:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='province')
    plt.title('Distribution of Policies by Province')
    plt.xticks(rotation=45)
    plt.savefig('scripts/province_bar.png')
    plt.close()
except Exception as e:
    print(f"Error in province bar plot: {e}")

# 4. Bivariate Analysis
try:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='totalpremium', y='totalclaims', hue='province', size='customvalueestimate', alpha=0.6)
    plt.title('TotalPremium vs TotalClaims by Province')
    plt.savefig('scripts/scatter_premium_claims.png')
    plt.close()
except Exception as e:
    print(f"Error in scatter plot: {e}")

# Correlation matrix
try:
    corr = df[numerical_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.savefig('scripts/correlation_matrix.png')
    plt.close()
except Exception as e:
    print(f"Error in correlation matrix: {e}")

# 5. Data Comparison
# Handle zero/negative TotalPremium
df['lossratio'] = np.where(df['totalpremium'] > 0, df['totalclaims'] / df['totalpremium'], np.nan)
try:
    loss_by_province = df.groupby('province')['lossratio'].mean().sort_values()
    plt.figure(figsize=(10, 5))
    loss_by_province.plot(kind='bar')
    plt.title('Average Loss Ratio by Province')
    plt.ylabel('Loss Ratio (Claims/Premium)')
    plt.xticks(rotation=45)
    plt.savefig('scripts/loss_ratio_province.png')
    plt.close()
except Exception as e:
    print(f"Error in loss ratio plot: {e}")

# 6. Outlier Detection
try:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[numerical_cols])
    plt.title('Box Plots for Outlier Detection')
    plt.savefig('scripts/box_plots.png')
    plt.close()
except Exception as e:
    print(f"Error in box plots: {e}")

# 7. Temporal Trends
if 'transactionmonth' in df.columns:
    try:
        df['transactionmonth'] = pd.to_datetime(df['transactionmonth'], errors='coerce')
        monthly_claims = df.groupby(df['transactionmonth'].dt.to_period('M'))['totalclaims'].sum()
        plt.figure(figsize=(12, 5))
        monthly_claims.plot()
        plt.title('Total Claims Over Time')
        plt.xlabel('Month')
        plt.ylabel('Total Claims')
        plt.savefig('scripts/temporal_trends.png')
        plt.close()
    except Exception as e:
        print(f"Error in temporal trends: {e}")

# 8. Key Insights
print("\n=== Key Insights ===")
print(f"Overall Loss Ratio: {df['lossratio'].mean():.2f}")
print(f"Highest Loss Ratio Province: {loss_by_province.idxmax()} ({loss_by_province.max():.2f})")
try:
    high_claim_vehicles = df.groupby('vehicletype')['totalclaims'].mean().sort_values(ascending=False).head(3)
    print("\nTop 3 Vehicle Types by Average Claims:")
    print(high_claim_vehicles)
except Exception as e:
    print(f"Error in vehicle type analysis: {e}")

# Save summary
with open('scripts/eda_summary.txt', 'w') as f:
    f.write("=== EDA Summary ===\n")
    f.write(f"Overall Loss Ratio: {df['lossratio'].mean():.2f}\n")
    f.write(f"Highest Loss Ratio Province: {loss_by_province.idxmax()} ({loss_by_province.max():.2f})\n")
    f.write("\nTop 3 Vehicle Types by Average Claims:\n")
    f.write(str(high_claim_vehicles))