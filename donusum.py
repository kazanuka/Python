import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.impute import SimpleImputer 
from sklearn import preprocessing
df = pd.read_csv("eksikveriler.csv")

#eksik verilerin düzenlenmesi



imputer = SimpleImputer(missing_values=np.nan,strategy='mean')#verileri impute etmek için bir kural tanımladık

yas = df.iloc[:,1:4].values #df değişkeninin 1-4 sütünları arasıdaki değerlerini yaş değişkenine atadık
                            #iloc => int location

imputer = imputer.fit(yas[:,1:4])#imputer kuralını ilgili sütunlar için hesapladık

yas[:,1:4] = imputer.transform(yas[:,1:4])#transform ile kuralı uyguladık ve geçersiz değerlerin yerine ortalama değeri yazdırdık


ulke = df.iloc[:,0:1].values

le = preprocessing.LabelEncoder()#le => label encoder (kategorik değerlere sayısal değerler atamak)
ulke[:,0] = le.fit_transform(df.iloc[:,0])

ohe = preprocessing.OneHotEncoder()#one hot encoder yöntemiyle kategorik değerleri binary
                                    #şeklinde gösterebildik
ulke = ohe.fit_transform(ulke).toarray()
#fit transform ile ulke sutununa işlemleri uyguladık
print(ulke)


