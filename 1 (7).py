import numpy as np
import streamlit as st
import datetime
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import openpyxl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
st.markdown(
    f"""
       <style>
       .stApp {{
           background-image: url("https://hilight.kapook.com/img_cms2/user/worawong/Hilight/1/spark.jpg");
           background-attachment: fixed;
           background-size: cover;
           /* opacity: 0.3; */
       }}
       </style>
       """,
    unsafe_allow_html=True
)

st.title(" นอนตอนนี้จะสดชื่อแค่ไหนกันนะ ")
option = st.selectbox(
    'ระบุวันที่เข้านอน (วันอาทิตย์=1,...,วันเสาร์=7)',
    ('1', '2', '3', '4', '5', '6', '7'))
left, right = st.columns(2)
a = left.time_input('ระบุเวลานอน',datetime.time(20, 0))
b = right.time_input('ระบุเวลาตื่น',datetime.time(5, 0))
time_b = (b.hour * 3600) + (b.minute * 60) + b.second
time_a = (a.hour * 3600) + (a.minute * 60) + a.second

df = pd.read_excel('time.xlsx')
x = pd.DataFrame(df)
time_wakeup = []
time_sleep = []
for i in range(len(x)):
    hour = x['เวลาตื่น'][i].hour
    minute = x['เวลาตื่น'][i].minute
    second = x['เวลาตื่น'][i].second
    t = (hour * 3600) + (minute * 60) + second
    time_wakeup.append(t)

for i in range(len(x)):
    hour = x['เวลานอน'][i].hour
    minute = x['เวลานอน'][i].minute
    second = x['เวลานอน'][i].second
    t = (hour * 3600) + (minute * 60) + second
    time_sleep.append(t)
y = df['ระดับความสดชื่น']
xx = pd.DataFrame({
        'x': time_wakeup,
        'y': time_sleep
            })

def load_data():
    return pd.read_excel('time.xlsx')

def save_model(model):
    joblib.dump(model, 'model.joblib')

def load_model():
    return joblib.load('model.joblib')

generateb = st.button('generate time.xlsx')
if generateb:
    st.write('generating "time.xlsx" ...')
    st.write(' ... done')

loadb = st.button('load time.xlsx')
if loadb:
    st.write('loading "file"')
    df = pd.read_excel('time.xlsx', index_col=0)
    st.write('... done')
    st.dataframe(df)

trainb = st.button('train')
if trainb:
    st.write('training model ...')
    df = pd.read_excel('time.xlsx', index_col=0)
    x_train,x_test,y_train,y_test = train_test_split(xx,y,test_size=0.2)
    model = LinearRegression()
    model.fit(x_train,y_train)
    st.write('... done')
    st.dataframe(df)
    save_model(model)

predictb = st.button('มาเช็คระดับความสดชื่นกันเลย')
if predictb:
    model = load_model()
    data_input = (time_b,time_a)
    data_input_array = np.asarray(data_input)
    data_array_reshape = data_input_array.reshape(1,-1)
    pred = model.predict(data_array_reshape)
    st.write(pred[0])