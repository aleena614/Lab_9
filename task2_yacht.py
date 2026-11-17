
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score

# Load the UCI Yacht Hydrodynamics dataset
yachts = pd.read_csv(
    "yacht_hydrodynamics.data",
    sep=r"\s+",
    header=None
)

# Select input feature and target
X = yachts[[5]]  # Froude number
y = yachts[6]    # Residuary resistance

# Display dataset information
print("Total experiments:", len(yachts))
print("Input feature: Froude number")
print("Target: Residuary resistance")

# Create Linear Regression model
linear_model = LinearRegression()

# Create Quadratic Regression model
quadratic_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression())
])

# Create five shuffled folds
folds = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Use negative RMSE scoring
scoring = "neg_root_mean_squared_error"

# Calculate positive RMSE for Linear Regression
linear_rmse = -cross_val_score(
    linear_model,
    X,
    y,
    cv=folds,
    scoring=scoring
)

# Calculate positive RMSE for Quadratic Regression
quadratic_rmse = -cross_val_score(
    quadratic_model,
    X,
    y,
    cv=folds,
    scoring=scoring
)

# Display fold RMSE values and mean
print("\nLinear Regression:")
print("Fold RMSEs:", linear_rmse.round(2))
print("Mean RMSE:", round(linear_rmse.mean(), 2))

print("\nQuadratic Regression:")
print("Fold RMSEs:", quadratic_rmse.round(2))
print("Mean RMSE:", round(quadratic_rmse.mean(), 2))

# Select the better model
print("\nModel Comparison:")

if linear_rmse.mean() < quadratic_rmse.mean():
    print("Linear Regression is the better model.")
else:
    print("Quadratic Regression is the better model.")
