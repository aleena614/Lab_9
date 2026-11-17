
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 1. Load dataset and remove duplicate rows
concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()

target = "Concrete compressive strength"

X = concrete.drop(columns=target)
y = concrete[target]

print("Total rows after removing duplicates:", len(concrete))


# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# 3. Create five shuffled folds
folds = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = "neg_root_mean_squared_error"


# 4. Linear Regression baseline
linear_model = LinearRegression()

linear_rmse = -cross_val_score(
    linear_model,
    X_train,
    y_train,
    cv=folds,
    scoring=scoring
)

baseline_mean = linear_rmse.mean()

print("\nLinear Regression Baseline:")
print("Fold RMSEs:", linear_rmse.round(2))
print("Mean CV RMSE:", round(baseline_mean, 2), "MPa")


# 5. Create Polynomial + Scaling + Ridge pipeline
ridge_model = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge())
])


# 6. Define parameter grid
param_grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}


# 7. Perform Grid Search
search = GridSearchCV(
    ridge_model,
    param_grid,
    cv=folds,
    scoring=scoring,
    refit=True
)

search.fit(X_train, y_train)


# 8. Display the three highest-ranked settings
results = pd.DataFrame({
    "Rank": search.cv_results_["rank_test_score"],
    "Degree": [
        params["poly__degree"]
        for params in search.cv_results_["params"]
    ],
    "Alpha": [
        params["ridge__alpha"]
        for params in search.cv_results_["params"]
    ],
    "Mean CV RMSE": -search.cv_results_["mean_test_score"]
})

top_three = results.sort_values("Rank").head(3)

print("\nTop 3 Grid Search Results:")
print(top_three.to_string(index=False))


# 9. Display best parameters
print("\nBest Parameters:", search.best_params_)
print("Best Mean CV RMSE:", round(-search.best_score_, 2), "MPa")


# 10. Evaluate the selected model on test data
test_predictions = search.predict(X_test)

test_rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

print("\nFinal Test Results:")
print("Test RMSE:", round(test_rmse, 2), "MPa")
print("Test MAE:", round(test_mae, 2), "MPa")


# 11. Plot actual vs predicted strength
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    test_predictions,
    label="Predicted values"
)

# Reference line y = x
minimum = min(y_test.min(), test_predictions.min())
maximum = max(y_test.max(), test_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    linestyle="--",
    label="Perfect prediction (y = x)"
)

plt.xlabel("Actual Concrete Strength (MPa)")
plt.ylabel("Predicted Concrete Strength (MPa)")
plt.title("Actual vs Predicted Concrete Strength")

plt.legend()
plt.grid(True)

# Save graph
plt.savefig(
    "task3_concrete.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
