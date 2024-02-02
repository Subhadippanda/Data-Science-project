import pandas as pd
df=pd.read_csv(r"C:\Users\SUBHADIP\Downloads\Admission_Prediction.csv")
df['GRE Score'].fillna(df['GRE Score'].mean(),inplace=True)
df['TOEFL Score'].fillna(df['TOEFL Score'].mean(),inplace=True)
df['University Rating'].fillna(df['University Rating'].mean(),inplace=True)
df1=df.copy()
df1.drop(columns=['Serial No.'],axis=1,inplace=True)
x=df.drop(columns=['Chance of Admit'])
y=df['Chance of Admit']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
arr=scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(arr,y,test_size=0.18,random_state=42,stratify=None)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
import pickle
filename='lr_for_admission.pkl'
pickle.dump(lr,open(filename,'wb'))
loaded_model=pickle.load(open('lr_for_admission.pkl','rb'))
result=loaded_model.score(x_test,y_test)
print(result)