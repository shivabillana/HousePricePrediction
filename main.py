# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,root_mean_squared_error

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
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# %%
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# %%
y_pred = regressor.predict(x_test)
print(y_pred)

# %%
mse = mean_squared_error(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)

# %%
print("Mean Squared Error :",mse)
print("Root Mean Squared Error :",rmse)

# %%
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.tight_layout()
plt.show()


