#1.kütüphaneler
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer #eksik verilerin doldurulmasi 
from sklearn import preprocessing #encodin(sayisal verilere donusum) işlemleri 
from sklearn.model_selection import train_test_split #train_test ayrimi
from sklearn.preprocessing import StandardScaler #scaling 
from sklearn.linear_model import LinearRegression #lineer reg kütüphanesi
import statsmodels.api as sm 


#2.veri önişleme

#2.1 veri yükleme

df = pd.read_csv("veriler.csv")

#2.2 KATEGORİK VERİLERİN SAYISAL VERİLERE DÖNÜŞÜMÜ(encoding) nominal/ordinal => numeric
boyY = df.iloc[:,1]
c = df.iloc[:,4:].values
le = preprocessing.LabelEncoder()
c[:,0] = le.fit_transform(df.iloc[:,4])

bagimsiz = df.iloc[:,2:4].values 
ulke = df.iloc[:,0:1].values


ulke[:,0] = le.fit_transform(df.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()


#2.3 VERİLERİN DF'LERE DÖNÜŞTÜRÜLMESİ
ulke = pd.DataFrame(data = ulke, index = range(22), columns = ['fr', 'tr' , 'us'])

bagimsiz = pd.DataFrame(data = bagimsiz, index = range(22), columns = ["kilo","yaş"])


erk = pd.DataFrame(data = c[:,0:1], index = range(22), columns = ["Erkek?"] )


#2.4 DF BİRLEŞTİRME İŞLEMİ
s = pd.concat([ulke,bagimsiz],axis = 1)
s2 = pd.concat([s,erk],axis =1)
x1 = s2.iloc[:,0:3]
x2 = s2.iloc[:,3:]
conX = pd.concat([x1,x2],axis = 1)

#2.5 VERİ KÜMELERİNİN TEST VE EĞİTİM OLARAK AYRILMASI

x_train,x_test,y_train,y_test = train_test_split(conX, boyY, test_size= 0.33, random_state = 0)

#2.6 ÖZNİTELİK ÖLÇEKLEME
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
tahmin = pd.DataFrame(data=tahmin,index = range(8),columns =["Boy"])

#4.GERİ ELEME
#4.1 ilk adım
X = np.append(arr = np.ones((22,1)).astype(int),values = conX, axis = 1)

liste = conX.iloc[:,[0,1,2,3,4,5]].values
liste = np.array(liste,dtype=float)

model = sm.OLS(boyY,liste).fit()
print(model.summary())

#4.2 ikinci adım
liste = conX.iloc[:,[0,1,2,3,5]].values
liste = np.array(liste,dtype=float)

model = sm.OLS(boyY,liste).fit()
print(model.summary())


