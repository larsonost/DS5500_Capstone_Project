import numpy as np
import pandas as pd
import os
import json
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def generate_image_features(model = DenseNet201())
    # load a list of val
    with open('/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_train2017.json', 'r') as f:
        df = json.load(f)
        df = df['annotations']

    # # Create a DataFrame from the image-caption pairs
    df = pd.DataFrame([i['image_id'] for i in df], columns=['image'])
    df = df.drop_duplicates(subset=['image'])

    # Create a feature extraction model by taking the output from the second last layer of the DenseNet201 model
    fe = Model(inputs=model.input, outputs=model.layers[-2].output)

    # Set the image size
    img_size = 224

    # Create a dictionary to store the image features
    features = {}
    part1 = df['image'].tolist()
    # Iterate over the unique image paths in your COCO dataset
    for image in tqdm(part1):
        img_name = '%012d.jpg' % image
        img_path = os.path.join('/kaggle/input/coco-2017-dataset/coco2017/train2017',img_name)
        
        # Load and preprocess the image using load_img and img_to_array functions
        img = load_img(img_path, target_size=(img_size, img_size))
        img = img_to_array(img)
        
        # Normalize the image pixel values to the range of [0, 1]
        img = img / 255.0
        
        # Expand the dimensions of the image array to match the model's input shape
        img = np.expand_dims(img, axis=0)
        
        # Extract the features by passing the preprocessed image through the feature extraction model (fe) using the predict method
        feature = fe.predict(img, verbose=0)
        
        # Store the extracted features in a dictionary with the image path as the key
        features[image] = feature.tolist()

    # save as a json
    with open('features.json', 'w') as json_file:
        json.dump(features, json_file)