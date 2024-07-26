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
ulke = ohe.fit_transform(ulke).toarray()#fit transform ile ulke sutununa işlemleri

#Yaş sütunda geçersiz değerlerin yerine ortalamayı yazarak bir sütun,
#ülke sütununda One Hot Encoding yaparak bir sütun oluşturduk.
#Bunların yanında bir de cinsiyet kümemiz vardı. Şu anda elimizde 3 adet
#veri kümesi bulunuyor. Bu veri kümelerini birleştirmeliyiz.

ulkeSonuc = pd.DataFrame(data = ulke, index = range(22), columns = ['fr', 'tr' , 'us'])
#print(sonuc)

yasSonuc = pd.DataFrame(data = yas, index = range(22), columns = ["boy","kilo","yaş"])

cinsiyet = df.iloc[:,4].values


cinsiyetSonuc = pd.DataFrame(data = cinsiyet, index = range(22), columns = ["cinsiyet"] )

#elimizde olan numpy matris dizilerini dataframe'e çevirdik
#şimdi bu 3 df'yi birleştireceğiz

s = pd.concat([ulkeSonuc,yasSonuc],axis = 1)

s2 = pd.concat([s,cinsiyetSonuc],axis = 1)
print(s2)


