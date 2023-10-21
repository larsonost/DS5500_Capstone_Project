import os
import warnings
import numpy as np
import json
import random
import string
import nltk
import collections
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pycocotools.coco import COCO
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from google.colab import drive

# Suppress warnings
warnings.filterwarnings('ignore')

# Connect to Google Drive for data access
drive.mount('/content/drive')

# Retrieve and extract required datasets
nltk.download('punkt')
#!unzip /content/drive/MyDrive/coco2017.zip -d /content/coco2017
#!unzip /content/drive/MyDrive/glove.6B.200d.txt -d /content/glove6b


# Paths for annotations and images
annotations_location = '/content/coco2017/annotations/captions_train2017.json'
image_directory = '/content/coco2017/train2017/'

# Retrieve captions from the given path
def fetch_annotations(annotations_location):
    with open(annotations_location, 'r') as annotation_file:
        data = json.load(annotation_file)
    return data

# Clean and format caption text
def format_caption_text(caption_text):
    punctuation_remover = str.maketrans('', '', string.punctuation)
    words = [word_item.lower().translate(punctuation_remover) for word_item in caption_text.split()]
    return ' '.join(words)

# Extract captions for each image
def extract_captions_for_images(data):
    caption_collections = collections.defaultdict(list)
    for entry in data['annotations']:
        formatted_caption = format_caption_text(entry['caption'])
        image_filename = f"{entry['image_id']:012}.jpg"
        full_image_path = os.path.join(image_directory, image_filename)
        caption_collections[full_image_path].append(formatted_caption)
    return dict(caption_collections)

# Get a subset of captions
def subset_captions(captions_collection, subset_fraction=0.5):
    items_list = list(captions_collection.items())
    random.shuffle(items_list)
    subset_count = int(len(items_list) * subset_fraction)
    return dict(items_list[:subset_count])

# Change paths to image IDs
def path_to_image_id_mapping(caption_collection):
    mapping = collections.defaultdict(list)
    for img_path, captions_list in caption_collection.items():
        img_identifier = os.path.basename(img_path).rstrip('.jpg')
        mapping[img_identifier] = captions_list
    return dict(mapping)
    

# Execute the sequence of functions
annotation_data = fetch_annotations(annotations_location)
captions_by_img = extract_captions_for_images(annotation_data)
selected_captions = subset_captions(captions_by_img)
mapped_captions = path_to_image_id_mapping(selected_captions)

# Record results
ids_of_images = list(mapped_captions.keys())
paths_of_images = list(selected_captions.keys())

# Pick a random image and its captions
random_image_path = random.choice(paths_of_images)
corresponding_captions = selected_captions[random_image_path]

# Display image using matplotlib
img = mpimg.imread(random_image_path)
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')  # Hide axis

# Create title with captions
caption_title = "\n".join(corresponding_captions)
plt.title(caption_title, fontsize=10)
plt.show()
