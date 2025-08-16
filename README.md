# House Price Prediction Project

### Project Overview

This project uses a simple linear regression model to predict house prices based on various features. The model is trained and evaluated using a provided dataset, and its performance is visualized with a scatter plot showing actual versus predicted prices.

---

### Files

- `housing_price_dataset.csv`: The dataset containing features such as `SquareFeet`, `Bedrooms`, `Bathrooms`, and `YearBuilt` used to predict the `Price`.
- The Python script: The code provided, which performs the following steps:
  - Data loading and inspection.
  - Splitting the data into training and testing sets.
  - Scaling the features using `StandardScaler`.
  - Training a `LinearRegression` model.
  - Making predictions and calculating the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
  - Generating a scatter plot to visualize the model's predictions.

---

### Key Metrics and Visualization

The model's performance is measured using the following metrics:

- **Mean Squared Error (MSE):** A measure of the average squared difference between the estimated values and the actual value.
- **Root Mean Squared Error (RMSE):** The square root of the MSE, which is a more interpretable metric as it is in the same units as the target variable (Price).

The generated graph, titled "Actual vs Predicted Prices", provides a visual representation of how well the model's predictions align with the actual prices. The red dashed line represents the ideal scenario, and the blue dots show the model's predictions for the test data.

---
