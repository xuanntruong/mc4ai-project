import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from deepface import DeepFace
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

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
def phannhom():
    with tab3:
        col1, col2 = st.columns(2)
    with col1:
        group = st.slider("Số nhóm", 2, 5)
    with col2:
        options = st.multiselect('Chọn đặc trưng', ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S-AVG', 'GPA'],
                                 ['S6', 'S10', 'GPA'], max_selections=3)

    X = df[options].values
    kmeans = KMeans(n_clusters=group, n_init='auto')
    kmeans.fit(X)
    labels = kmeans.predict(X)
    cluster = kmeans.cluster_centers_

    if len(options) == 3:
        fig = go.Figure(data=[go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers', marker=dict(color=labels))])
        fig.update_layout(
        scene=dict(
        xaxis=dict(title=options[0]),
        yaxis=dict(title=options[1]),
        zaxis=dict(title=options[2])
        )
        )
        st.plotly_chart(fig)
        colour= ["yellow", "red", "blue", "black", "purple"]
        for i in range(group):
            col3, col4 = st.columns(2)
            with col3:
                st.write(f"Nhóm {i+1}")
                st.write(f"GPA cao nhất{max(X[i])}")
                st.write(f"GPA cao nhất{min(X[i])}")
                st.write(f"GPA cao nhất{round(np.mean(X[i]), 2)}")
                filter = df[labels==i]
                filter = filter[[options[0], options[1], options[2]]]
                st.dataframe(filter)
            with col4:
                fig = go.Figure(data = [go.Scatter3d(x=filter[options[0]], y=filter[options[1]], z=filter[options[2]], mode='markers', marker=dict(color=colour[i]))])
                fig.update_layout(
                scene=dict(
                xaxis=dict(title=options[0]),
                yaxis=dict(title=options[1]),
                zaxis=dict(title=options[2])
                )
                )

            st.plotly_chart(fig)
    elif len(options)== 2:
        fig = go.Figure(data = [go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color=labels))],layout=go.Layout({'showlegend':True, 'xaxis_title':options[0], 'yaxis_title':options[1]}))
        st.plotly_chart(fig)
        for i in range(group):
            col5, col6 = st.columns(2)
            with col5:
                st.write(f"Nhóm {i+1}")
                st.write(f"GPA cao nhất{max(X[i])}")
                st.write(f"GPA cao nhất{min(X[i])}")
                st.write(f"GPA cao nhất{round(np.mean(X[i]), 2)}")
                filter = df[labels==i]
                filter = filter[[options[0], options[1]]]
                st.dataframe(filter)
            with col6:
                fig = go.Figure(data = [go.Scatter(x=filter[options[0]], y=filter[options[1]], mode='markers', marker=dict(color=labels))],layout=go.Layout({'xaxis_title':options[0], 'yaxis_title':options[1]}))
                st.plotly_chart(fig)
def phanloai():
    with tab4:
        dac_trung = st.multiselect("Chọn đặc trưng: ", ['S1', 'S2', 'S3', 'S4', 'S5', 'S6',
                                                        'S7', 'S8', 'S9', 'S10', 'GPA'],
                                                         ['S6', 'S10', 'GPA'], max_selections=3)
        if len(dac_trung) == 3:
            x = np.array(df.loc[:, [dac_trung[0], dac_trung[1]]].values)
            threshold = 5
            y = np.array(df[dac_trung[2]] >= threshold, dtype=int)  # Binary encoding
            model = LogisticRegression()
            model.fit(x, y)
            y_test_pred = model.predict(x)
            w = model.coef_[0]
            b = model.intercept_[0]
            w1, w2 = w
            xx, yy = np.meshgrid(np.linspace(x[:, 0].min(), x[:, 0].max(), 50),
                                 np.linspace(x[:, 1].min(), x[:, 1].max(), 50))
            if len(w) > 2:
                w3 = w[2]
                zz = (-w1 * xx - w2 * yy - b) / w3
            else:
                zz = (-w1 * xx - w2 * yy - b)
            zz = zz.reshape(xx.shape)

            fig = go.Figure(data=[
                go.Surface(x=xx, y=yy, z=zz, name='Decision Boundary'),
                go.Scatter3d(x=df[df[dac_trung[2]]< threshold][dac_trung[0]],
                             y=df[df[dac_trung[2]]< threshold][dac_trung[1]],
                             z=df[df[dac_trung[2]]< threshold][dac_trung[2]],
                             mode="markers",showlegend=False),
                go.Scatter3d(x=df[df[dac_trung[2]]>= threshold][dac_trung[0]],
                             y=df[df[dac_trung[2]]>= threshold][dac_trung[1]],
                             z=df[df[dac_trung[2]]>= threshold][dac_trung[2]],
                             mode="markers",showlegend=False)])
            fig.update_layout(scene=dict(
                xaxis_title=dac_trung[0],
                yaxis_title=dac_trung[1],
                zaxis_title=dac_trung[2]))
            st.plotly_chart(fig)
            accuracy = accuracy_score(y, y_test_pred)
            st.write("Độ chính xác: ", round(accuracy, 2))
        elif len(dac_trung)==2:
            threshold = 5  # Threshold value to determine the class labels
            x= np.where(df[dac_trung[0]]>= threshold,1,0).reshape(-1,1)
            y = np.where(df[dac_trung[1]] >= threshold, 1, 0)
            model = LogisticRegression()
            model.fit(x, y)
            w = model.coef_[0]
            b = model.intercept_[0]
            w1=w[0]
            x1=np.array([0,1])
            if len(w)>1:
                w2 = w[1]
                x2=-(w1/w2)*x1-b/w2
            else:
                x2=-(w1)*x1-b
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x[y == 0, 0], y=x[y == 0, 0], mode='markers'))
            fig.add_trace(go.Scatter(x=x[y == 1, 0], y=x[y == 1, 0], mode='markers'))
            fig.add_trace(go.Scatter(x=x1, y=x2, mode='lines'))
            fig.update_layout(
                title='Logistic Regression',
                xaxis_title=dac_trung[1],
                yaxis_title=dac_trung[0])
            st.plotly_chart(fig)

def xemdiem():
  with tab5:
      def cosine_similarity(vector_a, vector_b): # hàm tính cosine similarity
          dot_product = np.dot(vector_a, vector_b)

          norm_a = np.linalg.norm(vector_a)
          norm_b = np.linalg.norm(vector_b)

          cosine_similarity = dot_product / (norm_a * norm_b)

          return cosine_similarity

      def detectface(img): # lấy ra vector mặt
          embs = DeepFace.represent(img)
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
phanloai()
phannhom()
