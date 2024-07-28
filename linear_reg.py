#1.kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #train_test ayrimi
from sklearn.preprocessing import StandardScaler #scaling 
from sklearn.linear_model import LinearRegression #doğrusal regresyon için

#2.veri önişleme

#2.1 veri yükleme
df = pd.read_csv("satislar.csv")
#print(df)



aylar = df.iloc[:,0:1]
satislar = df.iloc[:,1:]
#iloc => int location
#print(aylar)


#2.6 VERİ KÜMELERİNİN TEST VE EĞİTİM OLARAK AYRILMASI

x_train,x_test,y_train,y_test = train_test_split(aylar, satislar, test_size= 0.33, random_state = 0)

#2.7 ÖZNİTELİK ÖLÇEKLEME

X_train = StandardScaler().fit_transform(x_train)
X_test = StandardScaler().fit_transform(x_test)
Y_train = StandardScaler().fit_transform(y_train)
Y_test = StandardScaler().fit_transform(y_test)


#3.1DOĞRUSAL REGRESYON MODEL İNŞASI
lr = LinearRegression()
lr.fit(x_train,y_train)

#3.2 MODELİN KULLANIMI
tahmin = lr.predict(x_test)

#3.3 MODELİN GÖRSELLEŞTİRİLMESİ
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)
