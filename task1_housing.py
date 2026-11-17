
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the UCI Real Estate dataset
homes = pd.read_csv("real_estate.csv")

# Convert distance into units of 100 meters
homes["distance_100m"] = (
    homes["X3 distance to the nearest MRT station"] / 100
)

# Select input feature and target
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Print slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict price at 500 meters
prediction = model.predict(
    pd.DataFrame({"distance_100m": [5]})
)

print("Predicted price at 500 meters:", prediction[0])

# Plot observations and fitted regression line
plt.scatter(X["distance_100m"], y, label="Actual observations")

plt.plot(
    X["distance_100m"],
    model.predict(X),
    color="red",
    label="Fitted regression line"
)

plt.xlabel("Distance to nearest MRT station (100 meters)")
plt.ylabel("House price per unit area")
plt.title("Housing Price Prediction Using Linear Regression")
plt.legend()
plt.grid(True)

# Save the graph
plt.savefig("task1_housing.png", dpi=300, bbox_inches="tight")

plt.show()
