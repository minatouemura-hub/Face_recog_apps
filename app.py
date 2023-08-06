import gradio as gr
from gradio import components
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, render_template, request, flash
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import PIL as pil

def preprocess_image(image:pil):
    # Resize the image to (224, 224) since that's the input size expected by the model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image_array = img_to_array(image)
    # Expand dimensions to match the model input shape (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    # Normalize the pixel values to be in the range [0, 1]
    image_array = image_array / 255.0
    return image_array

def predict(gr_input):
    model = load_model("bottleneck_fc_model.h5")
    preprocessed_input = preprocess_image(gr_input)
    pred = model.predict(preprocessed_input)
    labels = ["cool", "cute", "femini", "fresh"]
    confidence = {labels[i]:float(pred[0][i]) for i in range(len(labels))}
    return confidence

def main():
    inputs = components.Image(type="pil")
    outputs = components.Label(num_top_classes=4)
    demo = gr.Interface(fn=predict, inputs=inputs, outputs=outputs)
    demo.launch()

if __name__ == "__main__":
    main()
