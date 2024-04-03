import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


df = pd.read_csv('https://raw.githubusercontent.com/hanv81/mc4ai-project-template/main/py4ai-score.csv', index_col=None)
df.head()

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
df['CLASS-GROUP'] = df.apply(lop, axis=1)

tab1, tab2, tab3, tab4 = st.tabs(["Danh sách", "Biểu đồ", "Phân nhóm", "Phân loại"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender = None
        x = 10
        st.write("Giới tính")
        Nam = st.checkbox('Nam')
        Nu = st.checkbox('Nữ')
        if not Nam:
            gender = "F"
        if not Nu:
            gender = "M"
        else:
            x = 0        
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

    st.write('Lớp chuyên')


    col1, col2, col3, col4, col5  = st.columns(5)
    with col1:
        chuyen = []
        van = st.checkbox('Văn')
        toan = st.checkbox('Toán')
        if van:
            chuyen.append('Văn')
        if toan:
            chuyen.append('Toán')
    with col2:
        ly = st.checkbox('Lý')
        hoa = st.checkbox('Hóa')
        if ly:
            chuyen.append('Lý')
        if hoa:
            chuyen.append('Hóa')        
    with col3:
        anh = st.checkbox('Anh')
        tin = st.checkbox('Tin')
        if anh:
            chuyen.append('Anh')
        if tin:
            chuyen.append('Tin')
    with col4:
        sudia = st.checkbox('Sử Địa')
        trungnhat = st.checkbox('Trung Nhật')
        if sudia:
            chuyen.append('Sử Địa')
        if trungnhat:
            chuyen.append('Trung Nhật')
    with col5:
        th = st.checkbox('TH/SN')
        khac = st.checkbox('Khác')
        if th:
            chuyen.append('TH/SN')
        if khac:
            chuyen.append('Khác')
    st.write(chuyen[0])
    if x == 2:
        df1 = df[(df['CLASS-GROUP'].isin(chuyen)) & (df['CLASS'].str.endswith(buoi))]
    elif x == 1:
        df1 = df[(df['CLASS'].str.startswith(grade)) & (df['CLASS-GROUP'].isin(chuyen)) & (df['CLASS'].str.endswith(buoi))]
    elif x == 10:
        df1 = df[(df['GENDER'] == gender) & (df['CLASS'].str.startswith(grade)) & (df['CLASS-GROUP'].isin(chuyen)) & (df['CLASS'].str.endswith(buoi))]
    st.dataframe(df1)

with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
