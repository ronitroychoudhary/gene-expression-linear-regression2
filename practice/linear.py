"""
Linear Regression Tutorial for Gene Expression Data
A comprehensive guide for bioinformatics students

Author: Tutorial for practicing linear regression with gene expression
Dataset: We'll use a sample GEO dataset or generate synthetic data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("LINEAR REGRESSION TUTORIAL FOR GENE EXPRESSION DATA")
print("="*80)

# ============================================================================
# PART 1: DATA LOADING AND PREPARATION
# ============================================================================

print("\n" + "="*80)
print("PART 1: DATA LOADING")
print("="*80)

# For this tutorial, we'll create synthetic gene expression data
# In practice, you would load from GEO using GEOparse or pandas

print("\nCreating synthetic gene expression dataset...")
print("(In real analysis, you'd load from GEO using: df = pd.read_csv('GEO_series_matrix.txt', sep='\\t'))")

np.random.seed(42)

# Simulate 100 samples and 50 genes
n_samples = 100
n_genes = 50

# Create gene names
gene_names = [f'Gene_{i}' for i in range(1, n_genes + 1)]

# Generate correlated gene expression data (log2 normalized)
# Some genes will be correlated to make it realistic
base_expression = np.random.randn(n_samples, 10)
gene_data = np.zeros((n_samples, n_genes))

for i in range(n_genes):
    # Create correlations between genes
    gene_data[:, i] = (0.7 * base_expression[:, i % 10] + 
                       0.3 * np.random.randn(n_samples))

# Create target gene that depends on some predictor genes
target_gene = (0.5 * gene_data[:, 0] + 
               0.3 * gene_data[:, 5] + 
               0.2 * gene_data[:, 10] + 
               np.random.randn(n_samples) * 0.5)

# Create DataFrame
expression_df = pd.DataFrame(gene_data, columns=gene_names)
expression_df['Target_Gene'] = target_gene

# Add sample IDs and a clinical variable
expression_df.insert(0, 'Sample_ID', [f'Sample_{i}' for i in range(1, n_samples + 1)])
expression_df['Age'] = np.random.randint(30, 80, n_samples)
expression_df['Tumor_Size'] = np.random.uniform(1, 10, n_samples)

print(f"\nDataset shape: {expression_df.shape}")
print(f"Number of samples: {n_samples}")
print(f"Number of genes: {n_genes}")
print("\nFirst few rows:")
print(expression_df.head())

print("\nBasic statistics:")
print(expression_df[['Gene_1', 'Gene_2', 'Target_Gene', 'Age']].describe())

# ============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PART 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Check for missing values
print(f"\nMissing values: {expression_df.isnull().sum().sum()}")

# Distribution of target gene
print("\nTarget Gene Statistics:")
print(f"Mean: {expression_df['Target_Gene'].mean():.3f}")
print(f"Std: {expression_df['Target_Gene'].std():.3f}")
print(f"Skewness: {stats.skew(expression_df['Target_Gene']):.3f}")

# Visualize distributions and correlations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Distribution of target gene
axes[0, 0].hist(expression_df['Target_Gene'], bins=20, edgecolor='black')
axes[0, 0].set_title('Distribution of Target Gene')
axes[0, 0].set_xlabel('Expression Level')
axes[0, 0].set_ylabel('Frequency')

# 2. Q-Q plot for normality check
stats.probplot(expression_df['Target_Gene'], dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot - Target Gene')

# 3. Correlation heatmap (subset of genes)
genes_subset = ['Gene_1', 'Gene_6', 'Gene_11', 'Gene_16', 'Target_Gene']
corr_matrix = expression_df[genes_subset].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=axes[0, 2])
axes[0, 2].set_title('Correlation Matrix (Subset)')

# 4. Scatter plot: Gene_1 vs Target_Gene
axes[1, 0].scatter(expression_df['Gene_1'], expression_df['Target_Gene'], alpha=0.6)
axes[1, 0].set_xlabel('Gene_1 Expression')
axes[1, 0].set_ylabel('Target Gene Expression')
axes[1, 0].set_title('Gene_1 vs Target Gene')

# 5. Scatter plot: Age vs Target_Gene
axes[1, 1].scatter(expression_df['Age'], expression_df['Target_Gene'], alpha=0.6)
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Target Gene Expression')
axes[1, 1].set_title('Age vs Target Gene')

# 6. Box plot of expression levels
axes[1, 2].boxplot([expression_df['Gene_1'], expression_df['Gene_6'], 
                     expression_df['Target_Gene']])
axes[1, 2].set_xticklabels(['Gene_1', 'Gene_6', 'Target_Gene'])
axes[1, 2].set_ylabel('Expression Level')
axes[1, 2].set_title('Expression Level Distributions')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
print("\nEDA plots saved as 'eda_plots.png'")
plt.show()

# ============================================================================
# PART 3: SIMPLE LINEAR REGRESSION (ONE PREDICTOR)
# ============================================================================

print("\n" + "="*80)
print("PART 3: SIMPLE LINEAR REGRESSION")
print("="*80)

print("\nPredicting Target_Gene from Gene_1...")

# Prepare data
X_simple = expression_df[['Gene_1']]
y = expression_df['Target_Gene']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_simple, y, 
                                                      test_size=0.2, 
                                                      random_state=42)

# Fit model
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

# Make predictions
y_pred_train = model_simple.predict(X_train)
y_pred_test = model_simple.predict(X_test)

# Evaluate
print(f"\nModel Equation: Target_Gene = {model_simple.intercept_:.3f} + {model_simple.coef_[0]:.3f} * Gene_1")
print(f"\nTraining R²: {r2_score(y_train, y_pred_train):.3f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.3f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Regression line
axes[0].scatter(X_train, y_train, alpha=0.6, label='Training data')
axes[0].scatter(X_test, y_test, alpha=0.6, label='Test data', color='orange')
axes[0].plot(X_train, y_pred_train, 'r-', linewidth=2, label='Regression line')
axes[0].set_xlabel('Gene_1 Expression')
axes[0].set_ylabel('Target Gene Expression')
axes[0].set_title('Simple Linear Regression')
axes[0].legend()

# Residual plot
residuals_train = y_train - y_pred_train
axes[1].scatter(y_pred_train, residuals_train, alpha=0.6)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Predicted Values')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')

plt.tight_layout()
plt.savefig('simple_regression.png', dpi=300, bbox_inches='tight')
print("\nSimple regression plots saved as 'simple_regression.png'")
plt.show()

# ============================================================================
# PART 4: MULTIPLE LINEAR REGRESSION
# ============================================================================

print("\n" + "="*80)
print("PART 4: MULTIPLE LINEAR REGRESSION")
print("="*80)

print("\nUsing multiple genes as predictors...")

# Select features
feature_cols = ['Gene_1', 'Gene_6', 'Gene_11', 'Gene_16', 'Age', 'Tumor_Size']
X_multiple = expression_df[feature_cols]
y = expression_df['Target_Gene']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y, 
                                                      test_size=0.2, 
                                                      random_state=42)

# Fit model
model_multiple = LinearRegression()
model_multiple.fit(X_train, y_train)

# Predictions
y_pred_train = model_multiple.predict(X_train)
y_pred_test = model_multiple.predict(X_test)

# Evaluate
print(f"\nTraining R²: {r2_score(y_train, y_pred_train):.3f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}")

print("\nCoefficients:")
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model_multiple.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print(coef_df)

# Check for multicollinearity (VIF)
print("\nChecking Multicollinearity (Correlation Matrix):")
corr_features = X_train[feature_cols].corr()
print(corr_features.round(2))

# Diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Predicted vs Actual
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Values')
axes[0, 0].set_ylabel('Predicted Values')
axes[0, 0].set_title('Predicted vs Actual')

# 2. Residuals vs Fitted
residuals_test = y_test - y_pred_test
axes[0, 1].scatter(y_pred_test, residuals_test, alpha=0.6)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Fitted Values')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals vs Fitted')

# 3. Q-Q plot of residuals
stats.probplot(residuals_test, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot of Residuals')

# 4. Feature importance
axes[1, 1].barh(coef_df['Feature'], np.abs(coef_df['Coefficient']))
axes[1, 1].set_xlabel('Absolute Coefficient Value')
axes[1, 1].set_title('Feature Importance')

plt.tight_layout()
plt.savefig('multiple_regression.png', dpi=300, bbox_inches='tight')
print("\nMultiple regression plots saved as 'multiple_regression.png'")
plt.show()

# ============================================================================
# PART 5: REGULARIZED REGRESSION (Ridge, Lasso, Elastic Net)
# ============================================================================

print("\n" + "="*80)
print("PART 5: REGULARIZED REGRESSION")
print("="*80)

print("\nUsing more features (high-dimensional scenario)...")

# Use many genes as features
feature_cols_many = [f'Gene_{i}' for i in range(1, 31)]  # 30 genes
X_high_dim = expression_df[feature_cols_many]
y = expression_df['Target_Gene']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X_high_dim, y, 
                                                      test_size=0.2, 
                                                      random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train different models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Count non-zero coefficients
    if hasattr(model, 'coef_'):
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-5)
    else:
        n_nonzero = len(feature_cols_many)
    
    results.append({
        'Model': name,
        'Test R²': r2,
        'RMSE': rmse,
        'Non-zero Coeffs': n_nonzero
    })
    
    print(f"\n{name}:")
    print(f"  Test R²: {r2:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{len(feature_cols_many)}")

results_df = pd.DataFrame(results)
print("\n" + "="*40)
print("MODEL COMPARISON:")
print(results_df.to_string(index=False))
print("="*40)

# Visualize coefficients
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for idx, (name, model) in enumerate(models.items()):
    ax = axes[idx // 2, idx % 2]
    if hasattr(model, 'coef_'):
        coeffs = model.coef_
        ax.bar(range(len(coeffs)), coeffs)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=0.5)
        ax.set_title(f'{name} Coefficients')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Coefficient Value')

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=300, bbox_inches='tight')
print("\nRegularization comparison saved as 'regularization_comparison.png'")
plt.show()

# ============================================================================
# PART 6: CROSS-VALIDATION
# ============================================================================

print("\n" + "="*80)
print("PART 6: CROSS-VALIDATION")
print("="*80)

print("\nPerforming 5-fold cross-validation...")

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                cv=5, scoring='r2')
    print(f"\n{name}:")
    print(f"  CV R² scores: {cv_scores.round(3)}")
    print(f"  Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# ============================================================================
# FINAL RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("TUTORIAL COMPLETE!")
print("="*80)

print("""
KEY TAKEAWAYS:

1. DATA PREPARATION:
   - Always check for missing values and outliers
   - Visualize distributions and relationships
   - Consider log transformation for gene expression if needed

2. MODEL SELECTION:
   - Simple linear regression: Good for understanding relationships
   - Multiple regression: When you have multiple relevant predictors
   - Regularization (Ridge/Lasso): Essential when features > samples

3. DIAGNOSTICS TO CHECK:
   - Residual plots (should be random, centered at 0)
   - Q-Q plots (check normality of residuals)
   - Multicollinearity (correlation matrix, VIF)
   - Train vs Test performance (watch for overfitting)

4. NEXT STEPS FOR REAL GEO DATA:
   - Use GEOparse library: pip install GEOparse
   - Load data: geo = GEOparse.get_GEO(geo="GSE62944")
   - Extract expression matrix and phenotype data
   - Apply same workflow!

5. BIOLOGICAL INTERPRETATION:
   - Large positive coefficient = gene expression increases with predictor
   - Lasso can help with feature selection (sets irrelevant features to 0)
   - Always validate findings with biological knowledge

PRACTICE EXERCISES:
1. Try different alpha values for Ridge/Lasso
2. Add interaction terms between genes
3. Use PCA before regression for dimension reduction
4. Try predicting clinical outcomes instead of gene expression
5. Download real GEO data and repeat this analysis!
""")

print("\nAll plots saved. Good luck with your learning! 🧬")