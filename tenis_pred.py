#1.kütüphaneler
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer #eksik verilerin doldurulmasi 
from sklearn import preprocessing #encodin(sayisal verilere donusum) işlemleri 
from sklearn.model_selection import train_test_split #train_test ayrimi
from sklearn.linear_model import LinearRegression #lineer reg kütüphanesi
import statsmodels.api as sm 


#2.veri önişleme

#2.1 veri yükleme

df = pd.read_csv("odev_tenis.csv")

#2.2 KATEGORİK VERİLERİN SAYISAL VERİLERE DÖNÜŞÜMÜ(encoding) nominal/ordinal => numeric
hava = df.iloc[:,0].values
windy = df.iloc[:,3].values
y = df.iloc[:,4].values
diger = df.iloc[:,1:3]

le = preprocessing.LabelEncoder()
windy = le.fit_transform(windy)
hava = le.fit_transform(hava)
y = le.fit_transform(y)

ohe = preprocessing.OneHotEncoder()
hava = ohe.fit_transform(hava.reshape(-1,1)).toarray()


#2.3 VERİLERİN DF'LERE DÖNÜŞTÜRÜLMESİ
hava = pd.DataFrame(data = hava, index = range(14), columns = ['Overcast', 'Rainy' , 'Sunny'])

y = pd.DataFrame(data = y, index = range(14), columns = ["Played?"] )
windy = pd.DataFrame(data = windy, index = range(14),columns= ["Windy?"])

#2.4 DF BİRLEŞTİRME İŞLEMİ
s = pd.concat([hava,diger],axis = 1)
s2 = pd.concat([s,windy],axis = 1)
s3 = pd.concat([s2,y],axis =1)



#2.5 VERİ KÜMELERİNİN TEST VE EĞİTİM OLARAK AYRILMASI

x_train,x_test,y_train,y_test = train_test_split(hava, y, test_size= 0.33, random_state = 0)


#3.ÇOKLU LINEER REGRESYON

lr = LinearRegression()

lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)
tahmin = pd.DataFrame(data=tahmin,index = range(5),columns =["Played?"])

#4.GERİ ELEME
X = np.append(arr = np.ones((14,1)).astype(int),values = s, axis = 1)

liste = s2.iloc[:,[0,1,2]].values
liste = np.array(liste,dtype=float)

model = sm.OLS(y,liste).fit()
print(model.summary())
