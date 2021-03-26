import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from confidenceIntervals import *

# inserting data
score_data = pd.read_csv('data1516.csv')
x = score_data[['Home_3rdQ','Away_3rdQ']]
y = score_data.iloc[:,-1]

# train-test-split and scaling data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

# initialising model
model = LinearRegression()
model.fit(X_train, y_train)

y_predicted = [int(i) for i in list(model.predict(X_test).round())]

lower, middle, upper = find_ci(X_train, X_test, y_train)
arrays = return_arrays(lower,middle)

# # Record actual values on test set
predictions = pd.DataFrame(y_test)
predictions['prediction'] = y_predicted
predictions['confidence_intervals'] = arrays