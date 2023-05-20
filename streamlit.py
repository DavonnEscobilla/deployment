import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('final_model.h5')
    return model

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img.reshape(1, 64, 64, 3)  # Reshape image data to match model input shape
    prediction = model.predict(img_reshape)
    return prediction

def main():
    st.write("# Fashion MNIST")
    model = load_model()
    file = st.file_uploader("Choose a fashion item photo from your computer", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_names = [
            '0: T-shirt/top', '1: Trouser', '2: Pullover', '3: Dress',
            '4: Coat', '5: Sandal', '6: Shirt', '7: Sneaker', '8: Bag', '9: Ankle boot'
        ]
        string = "OUTPUT: " + class_names[np.argmax(prediction)]
        st.success(string)
