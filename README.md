# PRODIGY_ML_01
## Used Linear Regression model  to predict house prices using dataset from kaggle 
### Description
This task implemented a **Linear Regression model** to prdict house prices based on features such as square footage, number of bedrooms, and number of bathrooms.
### Dataset
- Source: Kaggle
- Files: `train.csv`
- Features:
  - `GrLivArea` (square footage)
  - `BedroomAbvGr` (number of bedrooms)
  - `FullBath` (number of bathrooms)
  - `SalePrice` (target price)
 ### Steps for it
 1. Load Dataset
2. Data Preprocessing (handle missing values, select columns, rename)
3. Train-Test Split
4. Train Linear Regression model
5. Predict house prices
6. Evaluate using  Mean Squared Error (MSE), Mean Absolute Error and  R² score
### how to run 
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

data=pd.read_csv("train.csv")
data.head()
print(data.columns)

#Preprocessing
data_req=data[['GrLivArea','BedroomAbvGr','FullBath','SalePrice']]
data_req.columns=['sqrt','Bedroom','Bathrooms','price']
data_req.dropna(inplace=True)
data_req.head()

x=data_req[['sqrt','Bedroom','Bathrooms']]
y=data_req['price']
#Datasplitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#model defining and fitting
model=LinearRegression()
model.fit(x_train,y_train)
#Prdection
y_pred=model.predict(x_test)
print("Mean_Squared_error:",mean_squared_error(y_test,y_pred))
print("Mean_Absolute_error:",mean_absolute_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))

##Results
Mean_Squared_Error: 2806426667.247853
Mean_Absolute_error: 35788.061292436294
R² Score: 0.6341189942328371
```
Author:
P.MANASWINI


