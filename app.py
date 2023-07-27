import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
from model import AutomateLablingModel

def load_model():
    with st.spinner('Model is loading...'):
        model = AutomateLablingModel()
    st.success('Done!')
    return model

def get_pil_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(opencv_image)
    return pil_image

def get_bar_percent(current_index, images_number):
    return int((current_index+1)*100/images_number)


st.title("Let's describe your fashion photos!")
csv = pd.DataFrame(data=[])


pil_images = []
with st.form('input form'):
    model = load_model()
    uploaded_files = st.file_uploader("Choose your images", type=["jpg", "png"], accept_multiple_files=True)

    if uploaded_files is not None:
        for file in uploaded_files:
            pil_image = get_pil_image(file)
            pil_images.append(pil_image)

    submit_button = st.form_submit_button('Submit')
    if submit_button:
        progress_text = "Building descriptions..."
        my_bar = st.progress(0, text=progress_text)
        to_save = []
        image_descriptions = []
        with st.container():
            for idx, pil_image in enumerate(pil_images):

                my_bar.progress(get_bar_percent(idx, len(pil_images)), text=progress_text)

                description = model.get_label(pil_image)

                st.image(pil_image, caption=description, width=300)
                to_save.append(st.checkbox('save', value=True, key=idx))
                image_descriptions.append(description)

        csv = pd.DataFrame({'image':pil_image, 'description':image_descriptions})
        csv['Save'] = to_save
        output_csv = csv[csv['Save'] == True]

st.download_button(
    label="Download descriptions in CSV",
    data=csv,
    file_name='descriptions.csv',
    mime='text/csv',
)

