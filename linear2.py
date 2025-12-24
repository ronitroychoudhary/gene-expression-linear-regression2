"""
Linear Regression with REAL GEO Data - Simple Tutorial
Dataset: GSE68465 (Lung Adenocarcinoma)
A step-by-step guide with detailed explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("LINEAR REGRESSION WITH REAL GEO DATA")
print("Dataset: GSE68465 (Lung Adenocarcinoma)")
print("="*80)

# ============================================================================
# STEP 1: DOWNLOAD AND LOAD GEO DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA FROM GEO")
print("="*80)

print("""
EXPLANATION:
- We're using GSE68465, a lung cancer gene expression dataset
- It has ~440 samples and thousands of genes
- We'll download it directly from GEO using pandas
- The data is already normalized (log2 transformed)
""")

try:
    # Download the series matrix file from GEO
    # This URL points to the processed, normalized data
    geo_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68465/matrix/GSE68465_series_matrix.txt.gz"
    
    print("\nDownloading data from GEO...")
    print("This may take 1-2 minutes depending on your connection...")
    
    # Read the file - it has metadata rows starting with '!'
    # We skip those and start from the actual data
    data = pd.read_csv(geo_url, sep='\t', skiprows=range(0, 71), compression='gzip')
    
    print(f"\n✓ Data loaded successfully!")
    print(f"Shape: {data.shape}")
    print(f"Columns: {len(data.columns)} (1 gene ID column + {len(data.columns)-1} samples)")
    
except Exception as e:
    print(f"\n✗ Error loading data: {e}")
    print("\nAlternative: You can manually download from:")
    print("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68465")
    print("Then load with: data = pd.read_csv('GSE68465_series_matrix.txt', sep='\\t', skiprows=range(0, 71))")
    raise

# ============================================================================
# STEP 2: EXPLORE AND CLEAN THE DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 2: EXPLORING THE DATA")
print("="*80)

print("""
EXPLANATION:
- First column is gene IDs (probe IDs from microarray)
- Each other column is a sample (patient)
- Values are gene expression levels (log2 normalized)
- We need to transpose: rows=samples, columns=genes
""")

# Look at the structure
print("\nFirst few rows and columns:")
print(data.iloc[:5, :5])

# Set gene IDs as index
data.set_index(data.columns[0], inplace=True)

# Transpose so samples are rows and genes are columns
# This is the standard format for machine learning
expression_df = data.T

print(f"\nAfter transposing:")
print(f"Shape: {expression_df.shape}")
print(f"Rows (samples): {expression_df.shape[0]}")
print(f"Columns (genes): {expression_df.shape[1]}")

# Check data quality
print(f"\nMissing values: {expression_df.isnull().sum().sum()}")
print(f"Data type: {expression_df.dtypes[0]}")

# Convert to numeric (sometimes strings sneak in)
expression_df = expression_df.apply(pd.to_numeric, errors='coerce')

# Remove any columns (genes) with missing values
expression_df = expression_df.dropna(axis=1)
print(f"\nAfter removing genes with missing values: {expression_df.shape}")

# ============================================================================
# STEP 3: SELECT GENES FOR ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: SELECTING GENES")
print("="*80)

print("""
EXPLANATION:
- We have thousands of genes, which is too many for simple regression
- Strategy: Select genes with high variance (more informative)
- We'll use top 100 most variable genes
- Then pick one as target, others as predictors
""")

# Calculate variance for each gene
gene_variance = expression_df.var().sort_values(ascending=False)

print(f"\nTop 10 most variable genes:")
print(gene_variance.head(10))

# Select top 100 most variable genes
n_top_genes = 100
top_genes = gene_variance.head(n_top_genes).index.tolist()
expression_subset = expression_df[top_genes]

print(f"\nSelected {n_top_genes} genes for analysis")
print(f"New shape: {expression_subset.shape}")

# ============================================================================
# STEP 4: SIMPLE LINEAR REGRESSION (1 gene predicts another)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: SIMPLE LINEAR REGRESSION")
print("="*80)

print("""
EXPLANATION:
- Let's predict Gene A from Gene B
- We'll use the 1st most variable gene as target
- And the 2nd most variable gene as predictor
- Question: Does Gene B expression predict Gene A expression?
""")

# Select target and predictor
target_gene = top_genes[0]
predictor_gene = top_genes[1]

print(f"\nTarget gene: {target_gene}")
print(f"Predictor gene: {predictor_gene}")

# Prepare X (predictor) and y (target)
X_simple = expression_subset[[predictor_gene]]  # Double brackets to keep as DataFrame
y = expression_subset[target_gene]

print(f"\nX shape: {X_simple.shape}")
print(f"y shape: {y.shape}")

# Split into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Create and train the model
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

# Get the equation: y = intercept + coefficient * x
intercept = model_simple.intercept_
coefficient = model_simple.coef_[0]

print(f"\nRegression Equation:")
print(f"{target_gene} = {intercept:.3f} + {coefficient:.3f} × {predictor_gene}")

# Make predictions
y_pred_train = model_simple.predict(X_train)
y_pred_test = model_simple.predict(X_test)

# Evaluate the model
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nModel Performance:")
print(f"Training R² = {train_r2:.3f} (How well model fits training data)")
print(f"Test R² = {test_r2:.3f} (How well model generalizes to new data)")
print(f"Test RMSE = {test_rmse:.3f} (Average prediction error)")

print("""
INTERPRETATION:
- R² ranges from 0 to 1 (higher is better)
- R² = 0.5 means model explains 50% of variance
- RMSE shows average error in same units as gene expression
""")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Scatter plot with regression line
axes[0].scatter(X_train, y_train, alpha=0.5, label='Training', s=30)
axes[0].scatter(X_test, y_test, alpha=0.5, label='Test', s=30, color='orange')
axes[0].plot(X_train, y_pred_train, 'r-', linewidth=2, label='Regression line')
axes[0].set_xlabel(f'{predictor_gene} Expression')
axes[0].set_ylabel(f'{target_gene} Expression')
axes[0].set_title('Simple Linear Regression')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Predicted vs Actual (test set)
axes[1].scatter(y_test, y_pred_test, alpha=0.6)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect prediction')
axes[1].set_xlabel('Actual Expression')
axes[1].set_ylabel('Predicted Expression')
axes[1].set_title(f'Predicted vs Actual (R² = {test_r2:.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Residuals (errors)
residuals = y_test - y_pred_test
axes[2].scatter(y_pred_test, residuals, alpha=0.6)
axes[2].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[2].set_xlabel('Predicted Values')
axes[2].set_ylabel('Residuals (Actual - Predicted)')
axes[2].set_title('Residual Plot')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_regression_geo.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved as 'simple_regression_geo.png'")
plt.show()

# ============================================================================
# STEP 5: MULTIPLE LINEAR REGRESSION (Multiple genes predict one gene)
# ============================================================================
print("\n" + "="*80)
print("STEP 5: MULTIPLE LINEAR REGRESSION")
print("="*80)

print("""
EXPLANATION:
- Now we use MULTIPLE genes to predict one target gene
- We'll use 10 genes as predictors
- This can capture more complex relationships
- Question: Can we improve prediction with more information?
""")

# Select target and multiple predictors
target_gene = top_genes[0]
predictor_genes = top_genes[1:11]  # Use genes 2-11 as predictors

print(f"\nTarget gene: {target_gene}")
print(f"Number of predictor genes: {len(predictor_genes)}")

# Prepare data
X_multiple = expression_subset[predictor_genes]
y = expression_subset[target_gene]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_multiple, y, test_size=0.2, random_state=42
)

# Standardize features (important for multiple regression!)
# This puts all genes on same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nData scaled to mean=0, std=1 for each gene")

# Train model
model_multiple = LinearRegression()
model_multiple.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = model_multiple.predict(X_train_scaled)
y_pred_test = model_multiple.predict(X_test_scaled)

# Evaluate
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nModel Performance:")
print(f"Training R² = {train_r2:.3f}")
print(f"Test R² = {test_r2:.3f}")
print(f"Test RMSE = {test_rmse:.3f}")

# Show feature importance (coefficients)
coef_df = pd.DataFrame({
    'Gene': predictor_genes,
    'Coefficient': model_multiple.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"\nFeature Importance (Coefficients):")
print(coef_df)

print("""
INTERPRETATION:
- Larger absolute coefficient = more important for prediction
- Positive coefficient = when this gene goes up, target goes up
- Negative coefficient = when this gene goes up, target goes down
""")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Predicted vs Actual
axes[0].scatter(y_test, y_pred_test, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2)
axes[0].set_xlabel('Actual Expression')
axes[0].set_ylabel('Predicted Expression')
axes[0].set_title(f'Multiple Regression\n(R² = {test_r2:.3f})')
axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y_test - y_pred_test
axes[1].scatter(y_pred_test, residuals, alpha=0.6)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Values')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')
axes[1].grid(True, alpha=0.3)

# Plot 3: Feature importance
axes[2].barh(coef_df['Gene'], np.abs(coef_df['Coefficient']))
axes[2].set_xlabel('Absolute Coefficient Value')
axes[2].set_title('Feature Importance')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multiple_regression_geo.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved as 'multiple_regression_geo.png'")
plt.show()

# ============================================================================
# STEP 6: LASSO REGRESSION (Automatic feature selection)
# ============================================================================
print("\n" + "="*80)
print("STEP 6: LASSO REGRESSION (FEATURE SELECTION)")
print("="*80)

print("""
EXPLANATION:
- Lasso automatically selects important genes
- It sets unimportant gene coefficients to ZERO
- This helps when you have many genes but only some matter
- Useful for finding biomarkers!
""")

# Use more predictor genes this time
predictor_genes_lasso = top_genes[1:31]  # 30 genes
X_lasso = expression_subset[predictor_genes_lasso]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X_lasso, y, test_size=0.2, random_state=42
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Lasso model
# alpha controls how aggressive the selection is
model_lasso = Lasso(alpha=0.01, random_state=42)
model_lasso.fit(X_train_scaled, y_train)

# Predictions
y_pred_test = model_lasso.predict(X_test_scaled)

# Evaluate
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nLasso Performance:")
print(f"Test R² = {test_r2:.3f}")
print(f"Test RMSE = {test_rmse:.3f}")

# Count selected features
n_selected = np.sum(np.abs(model_lasso.coef_) > 1e-5)
print(f"\nFeatures selected: {n_selected} out of {len(predictor_genes_lasso)}")

# Show selected genes
selected_genes = pd.DataFrame({
    'Gene': predictor_genes_lasso,
    'Coefficient': model_lasso.coef_
})
selected_genes = selected_genes[selected_genes['Coefficient'].abs() > 1e-5]
selected_genes = selected_genes.sort_values('Coefficient', key=abs, ascending=False)

print(f"\nSelected Genes (non-zero coefficients):")
print(selected_genes)

print("""
BIOLOGICAL INTERPRETATION:
- These selected genes are potential biomarkers
- They show strong correlation with target gene
- Could represent co-regulated genes or pathway members
- Further biological validation would be needed
""")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: All coefficients
axes[0].bar(range(len(model_lasso.coef_)), model_lasso.coef_)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=0.5)
axes[0].set_xlabel('Gene Index')
axes[0].set_ylabel('Coefficient')
axes[0].set_title(f'Lasso Coefficients\n({n_selected} non-zero out of {len(predictor_genes_lasso)})')
axes[0].grid(True, alpha=0.3)

# Plot 2: Predicted vs Actual
axes[1].scatter(y_test, y_pred_test, alpha=0.6)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2)
axes[1].set_xlabel('Actual Expression')
axes[1].set_ylabel('Predicted Expression')
axes[1].set_title(f'Lasso Prediction\n(R² = {test_r2:.3f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lasso_regression_geo.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved as 'lasso_regression_geo.png'")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TUTORIAL COMPLETE! 🎉")
print("="*80)

print("""
WHAT WE DID:
1. ✓ Downloaded real gene expression data from GEO (GSE68465)
2. ✓ Cleaned and prepared the data
3. ✓ Simple regression: 1 gene predicts another
4. ✓ Multiple regression: Multiple genes predict one gene
5. ✓ Lasso regression: Automatic feature selection

KEY CONCEPTS:
- R² score: How well the model explains variance (0-1, higher is better)
- RMSE: Average prediction error
- Coefficients: Show importance and direction of each gene
- Lasso: Automatically finds most important genes

NEXT STEPS:
1. Try different target genes
2. Add clinical data (age, stage, survival) if available
3. Use different alpha values in Lasso
4. Try Ridge regression (Ridge(alpha=1.0))
5. Interpret results biologically (pathway analysis, literature search)

QUESTIONS TO EXPLORE:
- Which genes are most predictive?
- Are they in the same pathway?
- Can you predict clinical outcomes instead of gene expression?
- What happens if you use top 20 vs top 100 genes?

Remember: In real research, you'd validate these findings in:
- Independent datasets
- Experimental validation
- Biological pathway analysis
- Literature review
""")

print("\n✓ All plots saved!")
print("✓ You now know how to do linear regression with real GEO data!")