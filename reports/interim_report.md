# Interim Report: Insurance Risk Analytics
**Author**: Yafet Mulugeta  
**Date**: 15 June 2025  
**Project**: 10 Academy AIM Week 3  

## Task 1: Git Setup and Exploratory Data Analysis

### Git Repository Setup
- Initialized repository: `https://github.com/Yafetm/-End-to-End-Insurance-Risk-Analytics-Predictive-Modeling`
- Created `task-1` branch
- Added folder structure: `.github/`, `src/`, `notebooks/`, `tests/`, `scripts/`, `reports/`
- Included `README.md`, `.gitignore`, `requirements.txt`, CI/CD workflow

### EDA Findings
1. **Data Overview**:
   - Analyzed `MachineLearningRating_v3.txt` (1,000,098 rows, Feb 2014 - Aug 2015)
   - Numerical features:
     - `TotalPremium`: Mean ~61.91, std ~230.28, min -782.58, max ~65,282.60
     - `TotalClaims`: Mean ~64.86, std ~2,384.08, min -12,002.41, max ~393,092.10
     - `CustomValueEstimate`: Mean ~225,531, std ~564,515, 779,642 missing (~78%)
   - Missing values:
     - `CustomValueEstimate`: 779,642
     - `NumberOfVehiclesInFleet`: 1,000,098
     - `CrossBorder`: 999,400
     - `WrittenOff`, `Rebuilt`, `Converted`: 641,901
     - `Bank`: 145,961
     - `AccountType`: 40,232
     - `NewVehicle`: 153,295
2. **Key Insights**:
   - Overall Loss Ratio: 0.35
   - Highest Loss Ratio Province: Gauteng (0.43)
   - Top Vehicle Types by Claims:
     - Heavy Commercial: 101.40
     - Medium Commercial: 76.32
     - Passenger Vehicle: 63.59
   - Outliers: Negative premiums/claims and extreme values detected
3. **Visualizations**:
   - Scatter plot: TotalPremium vs TotalClaims by Province (`scripts/scatter_premium_claims.png`)
   - Bar plot: Loss Ratio by Province (`scripts/loss_ratio_province.png`)
   - Box plots: Outlier detection (`scripts/box_plots.png`)
   - Additional: Histograms, correlation matrix, temporal trends

## Task 2: Data Version Control (DVC)
- Installed DVC and initialized repository.
- Tracked `data/MachineLearningRating_v3.txt` with DVC.
- Configured local remote storage at `C:/Users/hp/Desktop/dvc_remote`.
- Pushed dataset to DVC remote for reproducibility.

### Next Steps
- Task 3: A/B hypothesis testing
- Task 4: Predictive modeling

### Repository
[GitHub](https://github.com/Yafetm/-End-to-End-Insurance-Risk-Analytics-Predictive-Modeling)