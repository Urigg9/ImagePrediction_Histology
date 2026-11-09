import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt  

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

class_labels = ['tumor', 'stroma', 'complex', 'lympho', 'debris', 'mucosa', 'adipose', 'empty']

def main():
    model_path = 'model/resnet50_colorectal_histology.keras'
    model = tf.keras.models.load_model(model_path)

    all_images = []
    files = os.listdir('images')
    for file in files:
        if file.endswith((".png",".jpg",".tif", ".tiff", ".jpeg", ".JPG")):
            all_images.append(file)

    predict_image(all_images, model)

def predict_image(all_images, model):
    #Select the image
    print("List of images: \n")
    for index, img in enumerate(all_images):
        print(f"{index +1 }:{img} \n")
    
    while True:
        selected_img = int(input("\nEnter the number of the image you want to predict:  "))
        if selected_img >= 1 and selected_img <= len(all_images):
            selected_img = all_images[selected_img-1]
            break
        else:
            print(f"Error: Number out of range. Please enter a number between 1 and {len(all_images)}.")
     
    img_path = f'images/{selected_img}'

    #Load and process the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_processed = preprocess_input(img_array.astype("float32"))

    #Predict the image
    pred = model.predict(img_processed, verbose=0) 
    idx = np.argmax(pred)
    predicted_class = class_labels[idx]
    conf = np.max(pred) * 100

    # Displaying the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    text = f'Prediction: {predicted_class}, confidence: {conf:.2f}%'
    plt.xlabel(text, fontsize=14, fontweight='bold', labelpad=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
