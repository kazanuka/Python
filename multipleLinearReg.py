#1.kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer #eksik verilerin doldurulmasi 
from sklearn import preprocessing #encodin(sayisal verilere donusum) işlemleri 
from sklearn.model_selection import train_test_split #train_test ayrimi
from sklearn.preprocessing import StandardScaler #scaling 
from sklearn.linear_model import LinearRegression


#2.veri önişleme

#2.1 veri yükleme
df = pd.read_csv("veriler.csv")

#2.3 KATEGORİK VERİLERİN SAYISAL VERİLERE DÖNÜŞÜMÜ(encoding) nominal/ordinal => numeric
c = df.iloc[:,4:].values
le = preprocessing.LabelEncoder()
c[:,0] = le.fit_transform(df.iloc[:,4])


ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()



bagimsiz = df.iloc[:,1:4].values 
ulke = df.iloc[:,0:1].values


le = preprocessing.LabelEncoder()


ulke[:,0] = le.fit_transform(df.iloc[:,0])


ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()


#2.4 VERİLERİN DF'LERE DÖNÜŞTÜRÜLMESİ
ulke = pd.DataFrame(data = ulke, index = range(22), columns = ['fr', 'tr' , 'us'])

bagimsiz = pd.DataFrame(data = bagimsiz, index = range(22), columns = ["boy","kilo","yaş"])


erk = pd.DataFrame(data = c[:,0:1], index = range(22), columns = ["Erkek?"] )


#2.5 DF BİRLEŞTİRME İŞLEMİ
s = pd.concat([ulke,bagimsiz],axis = 1)
s2 = pd.concat([s,erk],axis =1)


#2.6 VERİ KÜMELERİNİN TEST VE EĞİTİM OLARAK AYRILMASI

x_train,x_test,y_train,y_test = train_test_split(s, erk, test_size= 0.33, random_state = 0)

#2.7 ÖZNİTELİK ÖLÇEKLEME
"""
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""
#3.ÇOKLU LINEER REGRESYON

lr = LinearRegression()

lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)
tahmin = pd.DataFrame(data=tahmin,index = range(8),columns =["Erkek?"])
