import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from deepface import DeepFace

df = pd.read_csv('https://raw.githubusercontent.com/hanv81/mc4ai-project-template/main/py4ai-score.csv', index_col=None)
df.head()

  #  Preprocess
for i in range(1, 11):
    a = 'S' + str(i)
    df[a].fillna(0, inplace=True)
df['BONUS'].fillna(0, inplace=True)
df['REG-MC4AI'].fillna('N', inplace=True)   # clear none


def lop(row):
    if 'CV' in row['CLASS']:
        return 'Văn'
    if 'CTI' in row['CLASS']:
        return 'Tin'
    if 'CL' in row['CLASS']:
        return 'Lý'
    if 'CH' in row['CLASS']:
        return 'Hóa'
    if 'CA' in row['CLASS']:
        return 'Anh'
    if 'CTR' in row['CLASS']:
        return 'Trung Nhật'
    if 'CS' in row['CLASS']:
        return 'Sử Địa'
    if 'TH' in row['CLASS'] or 'SN' in row['CLASS']:
        return 'TH/SN'
    if 'T' in row['CLASS']:
        return 'Toán'
    return 'Khác'
df['CLASS-GROUP'] = df.apply(lop, axis=1) # thêm phân loại lớp chuyên

  # tạo các tab
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Danh sách", "Biểu đồ", "Phân nhóm", "Phân loại", "xem điểm bằng gương mặt"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)  # tạo các ô lựa chọn

    with col1:
        gender = None
        x = 10
        st.write("Giới tính")
        Nam = st.checkbox('Nam')
        Nu = st.checkbox('Nữ')
        if Nam and Nu:
            x = 0
        elif Nu:
            gender = "F"
        elif Nam:
            gender = "M"     
    with col2:
        grade = st.radio("Khối lớp",("Tất cả", "Lớp 10", "Lớp 11", "Lớp 12"), index = None)
        if grade != "Tất cả" and x == 0:
            x = 1
        elif grade == "Tất cả" and x == 0:
            x = 2
        
        if grade == "Lớp 10":
            grade = '10'
        if grade == "Lớp 11":
            grade = '11'
        if grade == "Lớp 12":
            grade = '12'
    with col3: 
        phong = st.selectbox("phòng",("A114","A115"))
        if phong == "A114":
            phong = '114'
        else:
            phong = '115'

    with col4:
        buoi = st.selectbox("buổi",("Sáng","Chiều"))
        if buoi == "Sáng":
            buoi = "S"
        else:
            buoi = "C"

    st.write('Lớp chuyên') # tiêu đề


    col1, col2, col3, col4, col5  = st.columns(5) # các ô lựa chọn
    with col1:
        chuyen = ['Văn', 'Toán', 'Lý', 'Hóa', 'Anh', 'Tin', 'Sử Địa','Trung Nhật', 'TH/SN', 'Khác']
        van = st.checkbox('Văn')
        toan = st.checkbox('Toán')
        if not van:
            chuyen.remove('Văn')
        if not toan:
            chuyen.remove('Toán')
    with col2:
        ly = st.checkbox('Lý')
        hoa = st.checkbox('Hóa')
        if not ly:
            chuyen.remove('Lý')
        if not hoa:
            chuyen.remove('Hóa')        
    with col3:
        anh = st.checkbox('Anh')
        tin = st.checkbox('Tin')
        if not anh:
            chuyen.remove('Anh')
        if not tin:
            chuyen.remove('Tin')
    with col4:
        sudia = st.checkbox('Sử Địa')
        trungnhat = st.checkbox('Trung Nhật')
        if not sudia:
            chuyen.remove('Sử Địa')
        if not trungnhat:
            chuyen.remove('Trung Nhật')
    with col5:
        th = st.checkbox('TH/SN')
        khac = st.checkbox('Khác')
        if not th:
            chuyen.remove('TH/SN')
        if not khac:
            chuyen.remove('Khác')

    if x == 2:
        df1 = df[(df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi)) & (df['PYTHON-CLASS'].str.startswith(phong))]
    elif x == 1:
        df1 = df[(df['CLASS'].str.startswith(grade)) & (df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi)) & (df['PYTHON-CLASS'].str.startswith(phong))]
    elif x == 10 and grade != 'Tất cả':
        df1 = df[(df['GENDER'] == gender) & (df['CLASS'].str.startswith(grade)) & (df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi)) & (df['PYTHON-CLASS'].str.startswith(phong))]
    else:
        df1 = df[(df['GENDER'] == gender) & (df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi)) & (df['PYTHON-CLASS'].str.startswith(phong))]

    st.dataframe(df1)
with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)  

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
