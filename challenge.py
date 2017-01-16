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
x = data[:,0]
y = data[:,1]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

rand_entry = random.randrange(0,97)
predicted = (y[rand_entry])
actual = (body_reg.predict(x[rand_entry])[0][0])
print "The random entry selected is : " + str(rand_entry)
print "Brain weight is :- " + str(x[rand_entry])
print "Actual Body weight is :- " + str(actual)
print "Predicted body weight is :- " + str(predicted)
print "Percentage error in body weight :- " + str(abs(actual - predicted)*100/actual) + "%"

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
