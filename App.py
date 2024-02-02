import streamlit as st
import pandas as pd
import pickle
df=pd.read_csv("Admission_Prediction.csv")
df['GRE Score'].fillna(df['GRE Score'].mean(),inplace=True)
df['TOEFL Score'].fillna(df['TOEFL Score'].mean(),inplace=True)
df['University Rating'].fillna(df['University Rating'].mean(),inplace=True)
df1=df.copy()
df1.drop(columns=['Serial No.'],axis=1,inplace=True)
x=df.drop(columns=['Chance of Admit'])
y=df['Chance of Admit']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
st.title('Admission Data Science Project')
serial_no=st.number_input('serial_no',min_value=1.00,max_value=500.00)
Gre_score = st.number_input('Gre_score',min_value=290.00,max_value=340.00)
TOEFL_score=st.number_input('TOEFL_SCORE',min_value=92.00,max_value=120.00)
University_Rating=st.number_input('University_Rating',min_value=1,max_value=5)
SOP=st.number_input('SOP',min_value=1.00,max_value=5.00)
LOR=st.number_input('LOR',min_value=1.00,max_value=5.00)
CGPA=st.number_input('CGPA',min_value=6.8,max_value=9.92)
Research=st.number_input('Research',min_value=0,max_value=1)
user_input=[[serial_no,Gre_score,TOEFL_score,University_Rating,SOP,LOR,CGPA,Research]]
scaler.fit(x)
scaled_user_input=scaler.transform(user_input)
#st.write(user_input)
#st.write(scaled_user_input)
loaded_model=pickle.load(open('lr_for_admission.pkl','rb'))
result=loaded_model.predict(scaled_user_input)
if st.button("predict"):
    result_percentage=result*100
    st.write(result_percentage)
    st.header("your admission percentage is ",str(result_percentage))









