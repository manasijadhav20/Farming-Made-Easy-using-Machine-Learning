import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn.tree import DecisionTreeRegressor
data = pd.read_csv('G:/Crop_Prediction/static/Paddy.csv')

X = data.iloc[:,:-1].values
y = data.iloc[:, 3].values
depth = random.randrange(7,18)
x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(X,y,test_size=0.1,random_state=42, shuffle=True)
model =  DecisionTreeRegressor(max_depth=depth,random_state=0)
model.fit(x_training_set, y_training_set)
model2 =  DecisionTreeRegressor(max_depth=depth,random_state=0)
model2.fit(x_test_set, y_test_set)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
model_score = model.score(x_training_set,y_training_set)
model2_score = model2.score(x_test_set,y_test_set)
print("Coefficient of determination R^2 of the Test Set: ",model_score)
y_predicted = model.predict(x_test_set)
print("Coefficient of determination R^2 of the Train Set: ",model2_score)
y2_predicted = model2.predict(x_training_set)
print("Mean absolute error test set: %.2f"% mean_absolute_error(y_test_set, y_predicted))
print('Test Variance score: %.2f' % r2_score(y_test_set, y_predicted))
print("Mean absolute error Training Set: %.2f"% mean_absolute_error(y_training_set, y2_predicted))
print('Training Variance score: %.2f' % r2_score(y_training_set, y2_predicted))
accuracy = model.score(x_test_set,y_test_set)
print("Test Set Accuracy",accuracy)
from sklearn.model_selection import cross_val_predict
fig, ax = plt.subplots()
ax.scatter(y_test_set, y_predicted, edgecolors=(0, 0, 0))
ax.plot([y_test_set.min(), y_test_set.max()], [y_test_set.min(), y_test_set.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")
plt.show()


