import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
from model import AutomateLablingModel
from models_ensemble import EnsembleModel
import io

def get_pil_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    pil_image = Image.open(io.BytesIO(file_bytes))#Image.frombytes('RGB', (,128), image_data)
    return pil_image


def get_bar_percent(current_index, images_number):
    return int(current_index * 100 / images_number)


st.title("Let's describe your fashion photos!")

# load the model (only if not already loaded)
if 'model' not in st.session_state:
    with st.spinner('Model is loading...'):
        model = EnsembleModel()#AutomateLablingModel()
    st.session_state.model = model
    st.success('Done!')
else:
    model = st.session_state.model

pil_images = []
image_descriptions = []
output_csv = pd.DataFrame({'image': pil_images, 'description': image_descriptions})
submit_button = None


with st.form('input form'):
    uploaded_files = st.file_uploader("Choose your images", type=["jpg", "png"], accept_multiple_files=True)
    if uploaded_files is not None:
        for file in uploaded_files:
            pil_image = get_pil_image(file)
            pil_images.append(pil_image)

    submit_button = st.form_submit_button('Submit')

if submit_button:
    progress_text = "Building descriptions..."
    my_bar = st.progress(0, text=progress_text)

    for idx, pil_image in enumerate(pil_images):
        # redraw progress bar
        my_bar.progress(get_bar_percent(idx, len(pil_images)), text=progress_text)
        # build description
        description = model.get_label(pil_image)
        # draw image with the description
        st.image(pil_image, caption=description, width=400)
        image_descriptions.append(description)

    # Update the progress bar to 100% and display the final message
    my_bar.progress(100)
    st.success("Descriptions are generated!")

    output = pd.DataFrame({'description': image_descriptions})
    csv = output.to_csv().encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='descriptions.csv',
        mime='text/csv',
    )
