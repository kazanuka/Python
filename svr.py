"""
ÖNEMLİ HATIRLATMA
Support Vector Regression algoritmasında verilere scaling uygulamak önemlidir
zira SVR metodları çok büyük ya da çok küçük değerlere karşı hassastırlar. 
Fazla büyük ya da küçük değerler tahmin algoritmasında ciddi sapmalara 
yol açabilir. Bundan dolayı bu algoritma kullanılırken mutlaka StandardScaling() metodu
kullanılmalıdır.
02.08.24
"""
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# veri yukleme
veriler = pd.read_csv('maaslar.csv')


x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

#2.7 ÖZNİTELİK ÖLÇEKLEME

sc = StandardScaler()
sc2 = StandardScaler()
x_sc = sc.fit_transform(x)
y_sc = sc2.fit_transform(y)

#linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_sc,y_sc)

plt.scatter(x_sc,y_sc,color='red')
plt.plot(x_sc,lr.predict(x_sc), color = 'blue')
plt.title(label = "Linear")
plt.show()


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
lr2 = LinearRegression()
pr = PolynomialFeatures(degree = 4)
x_poly = pr.fit_transform(x_sc)

lr2.fit(x_poly,y_sc)
plt.scatter(x_sc,y_sc,color = 'red')
plt.plot(x_sc,lr2.predict(pr.fit_transform(x_sc)), color = 'blue')
plt.title(label ="Polynomial(Deg = 4)")
plt.show()


#support vector regression
from sklearn.svm import SVR
svr_lin = SVR(kernel = "linear",)
svr_rbf = SVR(kernel = "rbf")
svr_poly = SVR(kernel= 'poly',degree = 6)
svr_rbf.fit(x_sc,y_sc)
svr_lin.fit(x_sc,y_sc)
svr_poly.fit(x,y)

plt.scatter(x_sc, y_sc, color = "red",marker = "+")
plt.plot(x_sc,svr_rbf.predict(x_sc),color= "b")
plt.plot(x_sc,svr_lin.predict(x_sc),color= "c")

plt.title(label = "SVR Gaussian and Linear")
plt.legend()
plt.show()

plt.scatter(x, y, color = "k",marker = "+")
plt.plot(x,svr_poly.predict(x),color= "m")
plt.title(label = "SVR Poly (Deg= 6)")
plt.show()


#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dc = DecisionTreeRegressor(random_state=0)
r_dc.fit(x_sc,y_sc)
plt.scatter(x_sc,y_sc,color="r",marker = "+")
plt.plot(x_sc,r_dc.predict(x_sc),color = "b")

plt.title(label = "Decision Tree Regressor")
plt.show()

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
r_fs = RandomForestRegressor(n_estimators=10,random_state=0)
r_fs.fit(x,y.ravel())
plt.scatter(x,y,color = "r")
plt.plot(x,r_fs.predict(x),color = "b")
plt.title("Random Forest Regressor")
plt.show()


