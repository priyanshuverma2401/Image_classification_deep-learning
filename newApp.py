import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Function for MobileNetV2 ImageNet model

def mobilenetv2_imagenet():
    st.title("Image Classification with MobileNetV2")
    progress_message = st.empty()
    progress_message.text("Requesting to Upload Image")
    progress_bar = st.progress(0)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # progress_message = st.empty()
        progress_message.text("Processing image")
        
        image = Image.open(uploaded_file)
        
        # Convert image to RGB if it has an alpha channel
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        st.image(image, caption='Uploaded Image',  width=500)
        progress_bar.progress(30)
        progress_message.text("classifying Image")

        
        # Load MobileNetV2 model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
        progress_bar.progress(95)
        
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"It looks like this is a **{label}**, and the model is quite sure about it with a confidence level of **{score * 100 :.2f}%**.")
            # st.write(f"Looks like: {label}")
            # st.write(f"With confidence: {score * 100 :.2f}%")
        progress_message.text("done")
        progress_bar.progress(100)


# Function for CIFAR-10 model
def cifar10_classification():
    st.title("CIFAR-10 Image Classification")
    progress_message = st.empty()
    progress_message.text("Requesting to Upload Image")
    
    progress_bar = st.progress(0)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    
    if uploaded_file is not None:
        progress_message.text("Processing image")
        image = Image.open(uploaded_file)
        
        # Convert image to RGB if it has an alpha channel
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        st.image(image, caption='Uploaded Image', width=500)
        
        progress_bar.progress(30)
        progress_message.text("classifying Image")
        
        # Load CIFAR-10 model
        model = tf.keras.models.load_model('model111.h5')
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Preprocess the image
        img = image.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        progress_bar.progress(95)
        #It looks like this is a [CLASS], and the model is quite sure about it with a confidence level of [CONFIDENCE]
        st.write(f"It looks like this is a **{class_names[predicted_class]}**, and the model is quite sure about it with a confidence level of **{confidence * 100:.2f}%**")
        progress_message.text("done")
        progress_bar.progress(100)
        # st.write(f"Predicted Class: {class_names[predicted_class]}")
        # st.write(f"Confidence: {confidence * 100:.2f}%")

# Main function to control the navigation
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Model", ("CIFAR-10","MobileNetV2 (ImageNet)"))

    placeholder = st.sidebar.empty()
    placeholder.markdown("<div style='text-align: center; margin-top: 300px;'>"
                        "developed by <b>Priyanshu Kumar Verma</b>"
                        "</div>",
                        unsafe_allow_html=True)
    
    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()

if __name__ == "__main__":
    main()