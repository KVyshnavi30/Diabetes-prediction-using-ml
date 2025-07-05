import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn .metrics import accuracy_score
df=pd.read_csv('diabetes.csv')
print(df)
print(df.info())
print(df.shape)
df.isnull().sum()
df=df.drop_duplicates()
print("BMI",df[df['BMI']==0].shape[0])
print(df)
df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
print("BMI",df[df['BMI']==0].shape[0])
x=df.drop(columns='Outcome',axis=1)
print(x)
col=x.columns.to_list()
y=df['Outcome']
print(y)
sc=StandardScaler()# numbers to binary
x=sc.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model1=RandomForestClassifier()
model2=SVC()
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
y_pred1=model1.predict(x_test)
y_pred2=model2.predict(x_test)
res1=accuracy_score(y_test,y_pred1)
res2=accuracy_score(y_test,y_pred2)
print(res1)
print(res2)

import pickle
df=pd.read_csv('diabetes.csv')
with open('rf_model.pkl','wb') as rf:
    pickle.dump(model1,rf)
with open('scalar.pkl','wb') as s1:
    pickle.dump(sc,s1)
with open('columns.pkl','wb') as clm:
    pickle.dump(col,clm)