import streamlit as st
import tensorflow as tf
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from streamlit_card import card

app_mode = st.sidebar.selectbox("Select Page",["Home","Registered Patients"])
#Tensorflow Model Prediction
def model_prediction(Patients_x_ray_image):
    model = tf.keras.models.load_model("Tuberculosis_detection_model.keras")
    image = tf.keras.preprocessing.image.load_img(Patients_x_ray_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Main Page
if(app_mode=="Home"):
    st.header("TB Patients Recognition System")
    
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Normal','Tuberculosis']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
   


#About Project
elif(app_mode=="Registered Patients"):
    st.header("All Registered Patients")


