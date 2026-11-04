import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt  


class_labels = ['tumor', 'stroma', 'complex', 'lympho', 'debris', 'mucosa', 'adipose', 'empty']

def predict_image(image_path, model):
    # Usamos 'img' para no pisar el módulo 'image' de Keras
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_processed = preprocess_input(img_array.astype("float32"))

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

def main():
    model_path = 'model/resnet50_colorectal_histology.keras'
    model = tf.keras.models.load_model(model_path)

    image_path = 'images'
    img_name = input("Enter the image name (including .jpg / .png): \n Possible example: example_1.png, example_2.png, example_3.png, your_image.jpg\n> ")
    image_path = os.path.join(image_path, img_name)
    
    if os.path.exists(image_path):
        predict_image(image_path, model)
    else:
        print(f"Error: file coundn't be found in '{image_path}'")

if __name__ == "__main__":
    main()

