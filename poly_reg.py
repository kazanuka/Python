
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# veri yukleme
veriler = pd.read_csv('maaslar.csv')


x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


#linear regression

lr = LinearRegression()
lr.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lr.predict(X), color = 'blue')
plt.show()


#polynomial regression

pr = PolynomialFeatures(degree = 2)
x_poly = pr.fit_transform(X)

lr.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lr.predict(pr.fit_transform(X)), color = 'blue')
plt.show()


#4.dereceden polinom
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)


lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()



