#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# veri yukleme
veriler = pd.read_csv('maaslar.csv')


x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values


#linear regression

lr = LinearRegression()
lr.fit(x,y)

plt.scatter(x,y,color='red')
plt.plot(x,lr.predict(x), color = 'blue')
plt.show()


#polynomial regression

pr = PolynomialFeatures(degree = 2)
x_poly = pr.fit_transform(x)

lr.fit(x_poly,y)
plt.scatter(x,y,color = 'red')
plt.plot(x,lr.predict(pr.fit_transform(x)), color = 'blue')
plt.show()

#2.7 ÖZNİTELİK ÖLÇEKLEME

sc = StandardScaler()
x_sc = sc.fit_transform(x)
y_sc = sc.fit_transform(y)

#support vector regression

svr_lin = SVR(kernel = "linear")
svr_rbf = SVR(kernel = "rbf")
svr_poly = SVR(kernel= "poly",degree= 4 )
svr_rbf.fit(x_sc,y_sc)
svr_lin.fit(x_sc,y_sc)
svr_poly.fit(x_sc,y_sc)

plt.scatter(x_sc, y_sc, color = "red",marker = "o")
plt.plot(x_sc,svr_rbf.predict(x_sc),color= "blue")
plt.plot(x_sc,svr_lin.predict(x_sc),color= "c")
plt.plot(x_sc,svr_poly.predict(x_sc),color= "m")




plt.title(label = "SVR")
plt.show()
