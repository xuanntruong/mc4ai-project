import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from deepface import DeepFace
from PIL import Image

df = pd.read_csv('https://raw.githubusercontent.com/xuanntruong/mc4ai-project/main/dataset.csv', index_col=None)
df.head()

  #  Preprocess
for i in range(1, 11):
    a = 'S' + str(i)
    df[a].fillna(0, inplace=True)
df['BONUS'].fillna(0, inplace=True)
df['REG-MC4AI'].fillna('N', inplace=True)   # fill none


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
def danhsach():
    with tab1:
        col1, col2, col3, col4 = st.columns(4)  # tạo các ô lựa chọn
        x=-1
        run=False
        with col1:
            x = 10
            st.write("Giới tính")
            Nam = st.checkbox('Nam')
            Nu = st.checkbox('Nữ')
            if Nam and Nu:
                x=0
                run=True
            elif Nu:
                gender = "F"
                run=True
            elif Nam:
                gender = "M"
                run=True
        with col2:
            grade = st.radio("Khối lớp",("Tất cả", "Lớp 10", "Lớp 11", "Lớp 12"), index = None)
            if grade != "Tất cả" and x == 0:
                x = 1
            elif grade == "Tất cả" and x == 0:
                x = 2
            if grade== "Tất cả":
                grade= None
            if grade == "Lớp 10":
                grade = '10'
            elif grade == "Lớp 11":
                grade = '11'
            elif grade == "Lớp 12":
                grade = '12'
        with col3:
            a=0
            phong = st.selectbox("phòng",("Tất cả","A114","A115"))
            if phong == "A114":
                phong = '114'
            if phong =="Tất cả":
                phong = None
            else:
                phong = '115'

        with col4:
            buoi = st.multiselect("Buổi",("Sáng","Chiều"))
            if buoi == "Sáng":
                buoi = "S"
            if buoi == ["Sáng","Chiều"]:
                buoi = None
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
        if st.button("Run"):
            if x != -1 and run==True and len(chuyen) > 0:
                if x == 2:
                    if phong is None: 
                        if buoi is None: df1 = df[(df['CLASS-GROUP'].isin(chuyen))]
                        else: df1 = df[(df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi))]
                    else: 
                        if buoi is None: df1 = df[(df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.startswith(phong))]
                        else: df1 = df[(df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi)) & (df['PYTHON-CLASS'].str.startswith(phong))]
                elif x == 1:
                    if phong is None: 
                        if buoi is None: df1 = df[(df['CLASS'].str.startswith(grade)) & (df['CLASS-GROUP'].isin(chuyen))]
                        else: df1 = df[(df['CLASS'].str.startswith(grade)) & (df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi))]
                    else: 
                        if buoi is None: df1 = df[(df['CLASS'].str.startswith(grade)) & (df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.startswith(phong))]
                        else: df1 = df[(df['CLASS'].str.startswith(grade)) & (df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi)) & (df['PYTHON-CLASS'].str.startswith(phong))]
                else:
                    if phong is None:
                        if gender is None:
                            if grade is None:
                                if buoi is None: df1 = df[(df['CLASS-GROUP'].isin(chuyen))]
                                else: df1 = df[(df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi))] 
                            else:
                                if buoi is None: df1 = df[(df['CLASS-GROUP'].isin(chuyen)) & (df['CLASS'].str.startswith(grade))]
                                else: df1 = df[(df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi)) & (df['CLASS'].str.startswith(grade))]
                        else:
                            if grade is None:
                                if buoi is None: df1 = df[(df['GENDER'] == gender) & (df['CLASS-GROUP'].isin(chuyen))]
                                else: df1 = df[(df['GENDER'] == gender) & (df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi))]
                            else:
                                if buoi is None: df1 = df[(df['GENDER'] == gender) & (df['CLASS-GROUP'].isin(chuyen)) & (df['CLASS'].str.startswith(grade))]
                                else: df1 = df[(df['GENDER'] == gender) & (df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi)) & (df['CLASS'].str.startswith(grade))]
                    else: 
                        if grade is None:
                            if gender is None: 
                                if buoi is None: df1 = df[(df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.startswith(phong))]
                                else: df1 = df[(df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi)) & (df['PYTHON-CLASS'].str.startswith(phong))]
                            else:
                                if buoi is None: df1 = df[(df['GENDER'] == gender) & (df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.startswith(phong))]
                                else: df1 = df[(df['GENDER'] == gender) & (df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi)) & (df['PYTHON-CLASS'].str.startswith(phong))]
                        else:
                            if gender is None: 
                                if buoi is None: df1 = df[(df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.startswith(phong)) & (df['CLASS'].str.startswith(grade))]
                                else: df1 = df[(df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi)) & (df['PYTHON-CLASS'].str.startswith(phong)) & (df['CLASS'].str.startswith(grade))]
                            else:
                                if buoi is None: df1 = df[(df['GENDER'] == gender) & (df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.startswith(phong)) & (df['CLASS'].str.startswith(grade))]
                                else: df1 = df[(df['GENDER'] == gender) & (df['CLASS-GROUP'].isin(chuyen)) & (df['PYTHON-CLASS'].str.endswith(buoi)) & (df['PYTHON-CLASS'].str.startswith(phong)) & (df['CLASS'].str.startswith(grade))]
                st.write('Số HS: ', len(df1), '(',len(df1[df1['GENDER']== 'M']), 'Nam',len(df1[df1['GENDER']== 'F']),'Nữ' , ')')
                st.write('GPA: cao nhất', np.max(df1['GPA']), 'thấp nhất', np.min(df1['GPA']), 'trung bình', round(np.mean(df1['GPA']), 1))
                st.dataframe(df1)
            else:
                cola,colb,colc=st.columns(3)
                with colb:
                    st.write("Vui lòng chọn")
def bieudo():           
    with tab2:
        tab1_, tab2_ = st.tabs(["Số lượng học sinh", "Điểm"])
        with tab1_:
            classes = ['114-C', '114-S', '115-S', '115-C']
            fig1 = go.Figure(data=[go.Pie(labels=classes, values=[len(df[df['PYTHON-CLASS'] == i]) for i in classes])])
            fig1.update_layout(title="Theo lớp")
            st.plotly_chart(fig1)
            with st.expander("Kết luận:"):
                st.write('Cách sắp xếp số học sinh chia đều ra 2 phòng (A114 và A115), 2 buổi(sáng và chiều) là hợp lý, đáp ứng được nhu cầu của học sinh')

            gioitinh = ['F', 'M']
            fig2 = go.Figure(data=[go.Pie(labels=gioitinh, values=[len(df[df['GENDER'] == i]) for i in gioitinh])])
            fig2.update_layout(title="Theo giới tính")
            st.plotly_chart(fig2)
            with st.expander("Kết luận:"):
                st.write('Nam quan tâm đến AI nhiều hơn nữ')

            lopchuyen = ['Văn', 'Toán', 'Lý', 'Hóa', 'Anh', 'Tin', 'Sử Địa','Trung Nhật', 'TH/SN', 'Khác']
            fig3 = go.Figure(data=[go.Pie(labels=lopchuyen, values=[len(df[df['CLASS-GROUP'] == i]) for i in lopchuyen])])
            fig3.update_layout(title="Theo khối chuyên")
            st.plotly_chart(fig3)
            with st.expander("Kết luận:"):
                st.write('Khối chuyên Toán và lớp thường quan tâm tới AI nhiều nhất')
                st.write('Khối chuyên Trung Nhật ít quan tam tới AI nhất')

    with tab2_:
        sessions = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'GPA']
        us_session = st.radio('Điểm từng session', sessions, horizontal = True)
        for i in sessions:
            if us_session == i:
                figg = px.box(df,x = 'PYTHON-CLASS' ,y=i, color="GENDER")
                st.plotly_chart(figg)
                with st.expander("Kết luận:"):
                    st.write('Nhìn chung học sinh Nam học tốt hơn học sinh Nữ')
                figg2 = px.box(df,x = 'CLASS-GROUP' ,y=i)
                st.plotly_chart(figg2)
                with st.expander("Kết luận:"):
                    st.write('Khối chuyên tin học đều và giỏi nhất')
                    st.write('100% các học sinh khối chuyên toán, văn, lý, tin và lớp thường đậu')
        
        st.bar_chart(df, x="CLASS-GROUP", y="GPA", color="GENDER")
with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

def xemdiem():
  with tab5:
      def cosine_similarity(vector_a, vector_b): # hàm tính cosine similarity
          dot_product = np.dot(vector_a, vector_b)
  
          norm_a = np.linalg.norm(vector_a)
          norm_b = np.linalg.norm(vector_b)
  
          cosine_similarity = dot_product / (norm_a * norm_b)
  
          return cosine_similarity
    
      def detectface(img): # lấy ra vector mặt
          embs = DeepFace.represent(img, enforce_detection=False)
          face = np.array(embs[0]['embedding'])
          return face      
        
      img_file_buffer = st.camera_input("Take a picture")
      if img_file_buffer is not None:
          img = Image.open(img_file_buffer)
          img_array = np.array(img) # chuyển ảnh đã chụp sang dạng ma trận
  
          img1 = Image.open("Xtruong.jpg") # lấy ảnh để so sánh
          img1 = np.array(img1)

          st.write('Độ tự tin', cosine_similarity(detectface(img1), detectface(img_array)))
          if cosine_similarity(detectface(img1), detectface(img_array)) >= 0.5:
              st.dataframe(df.iloc[[-1]])
  

danhsach()
bieudo()
xemdiem()
