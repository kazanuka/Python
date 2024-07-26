import numpy as np
import pandas as pd
import matplotlib as plt

df = pd.read_csv("eksikveriler.csv")

#print(df)
#eksik verilerin düzenlenmesi

from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')#verileri impute etmek için bir kural tanımladık

yas = df.iloc[:,1:4].values #df değişkeninin 1-4 sütünları arasıdaki değerlerini yaş değişkenine atadık


imputer = imputer.fit(yas[:,1:4])#imputer kuralını ilgili sütunlar için hesapladık

yas[:,1:4] = imputer.transform(yas[:,1:4])#transform ile kuralı uyguladık ve geçersiz değerlerin yerine ortalama değeri yazdırdık

print(yas)



