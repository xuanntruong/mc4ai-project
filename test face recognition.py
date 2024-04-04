from PIL import Image
import numpy as np
import streamlit as st
from deepface import DeepFace

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    cosine_similarity = dot_product / (norm_a * norm_b)

    return cosine_similarity

img_file_buffer = st.camera_input("Take a picture")
if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    img_array = np.array(img)

    # st.write(type(img_array))
    # st.write((img_array.shape))
img1 = Image.open("Xtruong.jpg")
img1 = np.array(img1)

def detectface(img):
    embs = DeepFace.represent(img)
    face = np.array(embs[0]['embedding'])
    return face
# st.write((detectface(img)).shape)
# st.write((detectface(img_array)).shape)
st.write(cosine_similarity(detectface(img1), detectface(img_array)))


