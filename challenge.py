import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import random
import numpy as np
#read data
dataframe = pd.read_csv("challenge_dataset.txt", header = None)
x_values = dataframe[[0]]
y_values = dataframe[[1]]
data = dataframe.values

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#make prediction
prediction = body_reg.predict(x_values)
rand_entry = random.randrange(0,97)
x_sample = x_values.iloc[rand_entry]
y_sample = (y_values.iloc[rand_entry])
prediction_sample = prediction[rand_entry]
error = abs(prediction_sample - y_sample)
percentage_error = error*100/y_sample
print "The random entry selected is : %f" % rand_entry
print "Brain weight is :- %f" % x_sample
print "Actual Body weight is :- %f" % y_sample
print "Predicted body weight is :- %f" % prediction_sample
print "Percentage error in body weight :- %f" % percentage_error

#visualize results
plt.scatter(x_values, y_values)
plt.scatter(x_sample, y_sample, color='red', s=95)
plt.scatter(x_sample, prediction_sample, color='green', s=75)
plt.plot(x_values, prediction)
plt.show()
