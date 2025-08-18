# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# %%
dataset = pd.read_csv('housing_price_dataset.csv')
dataset.head()

# %%
dataset.isnull().sum()

# %%
x = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,-1].values

# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

# %%
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

# %%
regressor = LinearRegression()
regressor.fit(x_train_sc,y_train)

# %%
y_pred_sc = regressor.predict(x_test_sc)
print(y_pred_sc)

# %%
mse = mean_squared_error(y_test,y_pred_sc)
rmse = np.sqrt(mse)

# %%
print("Mean Squared Error :",mse)
print("Root Mean Squared Error :",rmse)

# %%
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_sc, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
print(regressor.coef_)
print(regressor.intercept_)

# %%
# %% Print regression equation
print("Regression Equation:")
equation = "y = {:.2f}".format(regressor.intercept_)
for i, coef in enumerate(regressor.coef_):
    equation += " + ({:.2f} * x{})".format(coef, i+1)
print(equation)


