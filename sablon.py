

#1.kütüphaneler
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.impute import SimpleImputer #eksik verilerin doldurulmasi 
from sklearn import preprocessing #encodin(sayisal verilere donusum) işlemleri 
from sklearn.model_selection import train_test_split #train_test ayrimi
from sklearn.preprocessing import StandardScaler #scaling 

#2.veri önişleme

#2.1 veri yükleme
df = pd.read_csv("eksikveriler.csv")

#2.2 EKSIK VERILERIN DÜZENLENMESI   

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

yas = df.iloc[:,1:4].values 
#iloc => int location
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])


#2.3 KATEGORİK VERİLERİN SAYISAL VERİLERE DÖNÜŞÜMÜ(encoding) nominal/ordinal => numeric

ulke = df.iloc[:,0:1].values

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(df.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

#2.4 VERİLERİN DF'LERE DÖNÜŞTÜRÜLMESİ
ulkeSonuc = pd.DataFrame(data = ulke, index = range(22), columns = ['fr', 'tr' , 'us'])
#print(sonuc)

yasSonuc = pd.DataFrame(data = yas, index = range(22), columns = ["boy","kilo","yaş"])

cinsiyet = df.iloc[:,4].values

cinsiyetSonuc = pd.DataFrame(data = cinsiyet, index = range(22), columns = ["cinsiyet"] )


#2.5 DF BİRLEŞTİRME İŞLEMİ
s = pd.concat([ulkeSonuc,yasSonuc],axis = 1)
s2 = pd.concat([s,cinsiyetSonuc],axis =1)

#2.6 VERİ KÜMELERİNİN TEST VE EĞİTİM OLARAK AYRILMASI

x_train,x_test,y_train,y_test = train_test_split(s, cinsiyetSonuc, test_size= 0.33, random_state = 0)

#2.7 ÖZNİTELİK ÖLÇEKLEME

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_train)


