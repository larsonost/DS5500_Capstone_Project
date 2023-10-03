import os
import nltk
import pandas as pd
from collections import Counter
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.nn.functional import cross_entropy
import matplotlib.pyplot as plt
import random
from torch.utils.data import Subset
from torch.nn.utils.rnn import pack_padded_sequence
from google.colab import drive
drive.mount('/content/drive')


nltk.download('punkt')
!unzip /content/drive/MyDrive/coco2017.zip -d /content/coco2017

# Update paths as necessary
annFile = '/content/coco2017/annotations/captions_train2017.json'
img_dir = '/content/coco2017/train2017'

coco = COCO(annFile)

def plot_random_image_with_captions(coco, img_dir):
    # Get all the image ids from the COCO dataset
    img_ids = list(coco.imgs.keys())
    
    # Randomly select an image
    random_img_id = random.choice(img_ids)
    img_data = coco.loadImgs(random_img_id)[0]
    img_path = os.path.join(img_dir, img_data['file_name'])
    
    # Load and plot the image
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')  # turn off axis numbers
    plt.title("Random Image")
    plt.show()
    
    # Fetch the corresponding captions
    annIds = coco.getAnnIds(imgIds=random_img_id)
    anns = coco.loadAnns(annIds)
    print("Captions:")
    for i, ann in enumerate(anns, 1):
        print(f"{i}. {ann['caption']}")
    
# Call the function
plot_random_image_with_captions(coco, img_dir)