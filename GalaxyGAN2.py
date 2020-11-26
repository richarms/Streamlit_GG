from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.enable_eager_execution()

import streamlit as st
import pandas as pd
import numpy as np


import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os

import PIL
from tensorflow.keras import layers
import time

from IPython import display


import_path = './models/'
generator = tf.keras.models.load_model(import_path + 'GGAN_generator.h5')


st.title('Galaxy GAN')
st.subheader("A Generative Adverserial Network trained on the FRICAT, FRIICAT, and NVSS sample of extended radio galaxies. Click the button to perform inference on this model.")

st.subheader("Button")
st.subheader("")
w1 = st.button("Generate Radio Galaxy from GAN Model")
st.write(w1)

if w1:
    #st.write("Interactive Streamlit App")
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    i=generated_image[0, :, :, 0].numpy()
    fig = plt.imshow(i, cmap = 'magma', aspect='equal')
    plt.show()
    st.pyplot(fig=fig)
    st.image(i, clamp=True, use_column_width=True)
