import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
s=pickle.load(open('scalar.pkl','rb'))
m=pickle.load(open('rf_model.pkl','rb'))
c=pickle.load(open('columns.pkl','rb'))
new_data={
    'Pregnancies':int(input("Enter Pregnancies : ")),
    'Glucose':int(input("Enter Glucose : ")),
    'BloodPressure':int(input("Enter BloodPressure : ")),
    'SkinThickness':int(input("Enter SkinThickness : ")),
    'Insulin':int(input("Enter Insulin : ")),
    'BMI':float(input("Enter BMI : ")),
    'DiabetesPedigreeFunction':float(input("Enter DiabetesPedigreeFunction : ")),
    'Age':int(input("Enter Age : "))
}
new_df = pd.DataFrame([new_data])[c]
scalar=s.transform(new_df)
ans=m.predict(scalar)
print(ans)