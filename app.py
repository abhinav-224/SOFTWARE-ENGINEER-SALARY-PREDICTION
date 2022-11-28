import streamlit as st
import pickle
import numpy as np
rf=pickle.load(open('rf.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

st.title("Salary Predictor")


comp=st.selectbox('Company Name',df['Company Name'].unique())

title=st.selectbox('Title',df['Job Title'].unique())

loc= st.selectbox('Location',df['Location'].unique())

rat=st.selectbox('Rating',df['Rating'].unique())

sal=st.selectbox('Average people working on a project ',df['Salaries Reported'].unique())

query=[[0]*2]*2
if st.button('Predict Salary'):
    query= np.array([rat,sal])

    query=query.reshape(1,2)

# st.title(rf.predict(2.303**query))

# ans= 2.303 ** query
# st.title(ans)
ans=2.303**rf.predict(query)
st.title(ans)

