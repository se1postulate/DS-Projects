import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

from sklearn.model_selection import ShuffleSplit

def Encoder(df):
          columnsToEncode = list(df.select_dtypes(include=['category','object']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
          return df
      
        
alldata = pd.read_csv("data/diamonds.csv")
#print(alldata)

alldata = Encoder(alldata)


print("________________________________________________________")
df_0 = alldata[alldata.cut==0]
df_1=alldata[alldata.cut==1]
df_2 = alldata[alldata.cut==2]
df_3=alldata[alldata.cut==3]
df_4=alldata[alldata.cut==4]
print("before changing")

print(len(df_0))
print(len(df_1))
print(len(df_2))
print(len(df_3))
print(len(df_4))
df_0_new = resample(df_0,replace=True,n_samples=(len(df_1.index))) # reproducible results
df_2_new = resample(df_2,replace=True,n_samples=(len(df_1.index)))
df_3_new = resample(df_3,replace=True,n_samples=(len(df_1.index)))
df_4_new = resample(df_4,replace=True,n_samples=(len(df_1.index)))
print("after changing")
print(len(df_0_new))
print(len(df_2_new))
print(len(df_1))
print(len(df_3_new))
print(len(df_4_new))
print("dataFrame")
alldata2=pd.concat([df_0_new,df_2_new])
alldata2=pd.concat([alldata2,df_1])
alldata2=pd.concat([alldata2,df_3_new])
alldata2=pd.concat([alldata2,df_4_new])
print("________________________________________________________")
#print(); print(alldata)

X=alldata2.iloc[:,[0,1,2,3,4,5,6,7,8]];
Y=alldata2.iloc[:,9];

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.55,shuffle=False,random_state=4906)

classifier_knn = KNeighborsClassifier(n_neighbors = 1)
classifier_knn.fit(X_train,y_train)

print(classifier_knn)
res=classifier_knn.predict(X_test)
print(res)
print(confusion_matrix(y_test, res))
print(classification_report(y_test, res))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15, 7), random_state=1)
clf.fit(X_train,y_train)
res1=classifier_knn.predict(X_train)
print(res)
print(confusion_matrix(y_train, res1))
print(classification_report(y_train, res1))
