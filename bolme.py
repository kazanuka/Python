

#1.kütüphaneler
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.impute import SimpleImputer 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

#2.veri önişleme
#2.1 veri yükleme
df = pd.read_csv("eksikveriler.csv")

#2.2 EKSIK VERILERIN DÜZENLENMESI   

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')#verileri impute etmek için bir kural tanımladık

yas = df.iloc[:,1:4].values #df değişkeninin 1-4 sütünları arasıdaki değerlerini yaş değişkenine atadık
                            #iloc => int location
imputer = imputer.fit(yas[:,1:4])#imputer kuralını ilgili sütunlar için hesapladık
yas[:,1:4] = imputer.transform(yas[:,1:4])#transform ile kuralı uyguladık ve geçersiz değerlerin yerine ortalama değeri yazdırdık


#2.3 KATEGORİK VERİLERİN SAYISAL VERİLERE DÖNÜŞÜMÜ(encoding) nominal/ordinal => numeric

ulke = df.iloc[:,0:1].values

le = preprocessing.LabelEncoder()#le => label encoder (kategorik değerlere sayısal değerler atamak)
ulke[:,0] = le.fit_transform(df.iloc[:,0])
#önce le kullandık, sonra le uygulanmış verilere ohe uyguladık
ohe = preprocessing.OneHotEncoder()#one hot encoder yöntemiyle kategorik değerleri binary
                                    #şeklinde gösterebildik
ulke = ohe.fit_transform(ulke).toarray()#fit transform ile ulke sutununa işlemleri
print(ulke)
#Yaş sütunda geçersiz değerlerin yerine ortalamayı yazarak bir sütun,
#ülke sütununda One Hot Encoding yaparak bir sütun oluşturduk.
#Bunların yanında bir de cinsiyet kümemiz vardı. Şu anda elimizde 3 adet
#veri kümesi bulunuyor. Bu veri kümelerini birleştirmeliyiz.

#VERİLERİN DF'LERE DÖNÜŞTÜRÜLMESİ
ulkeSonuc = pd.DataFrame(data = ulke, index = range(22), columns = ['fr', 'tr' , 'us'])
#print(sonuc)

yasSonuc = pd.DataFrame(data = yas, index = range(22), columns = ["boy","kilo","yaş"])

cinsiyet = df.iloc[:,4].values

cinsiyetSonuc = pd.DataFrame(data = cinsiyet, index = range(22), columns = ["cinsiyet"] )

#elimizde olan numpy matris dizilerini dataframe'e çevirdik
#şimdi bu 3 df'yi birleştireceğiz

s = pd.concat([ulkeSonuc,yasSonuc],axis = 1)#concatenete => birleştirmek
#elimizde bağımsız değişkenleri yani milliyet,boy kilo yas bir df, cinsiyet bir df olacak şekilde tutuyoruz.
s2 = pd.concat([s,cinsiyetSonuc],axis =1)

#VERİ KÜMELERİNİN AYRILMASI

x_train,x_test,y_train,y_test = train_test_split(s, cinsiyetSonuc, test_size= 0.33, random_state = 0)
#milliyet, boy, kilo ve yaş bağımsız; cinsiyet bağımlı değişken. yani aslında cinsiyet tahmin veriseti.

#ÖZNİTELİK ÖLÇEKLEME
#bu method kullanmış olduğumuz sayısal değerlerin birbirine daha yakın olacak şekilde ölçeklenmesine olanak sağladı
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_train)


