import json
from os.path import join, dirname
from os import environ
from dataset import *
import numpy as npy
from tqdm import tqdm
import re


datasets = loadDatasetRaw()

'''
train_data = datasets[0]
train_labels = datasets[1]
painting = train_data[0][0]
p2 = train_data[0][1]
score = train_labels[0]
print(score)
print("\n##########################################")
print("Corrupted painting: " + str(painting.imageFilename()))
print("Theme: " + painting.theme)
print("Title: " + painting.title)
print("Artist: " + painting.artist)
print("Style: " + painting.style)
print("Genre: " + painting.genre)
print("Wiki URL: " + painting.wikiURL)
print("Image URL: " + painting.imageURL)
print("##########################################\n")
'''
img_titles = {}
for i in range(len(datasets)):
    if i % 2 == 1:
        continue
    data = datasets[i]
    for i in tqdm(range(1200)):
        for painting in data[i]:
            img_titles[painting.imageFilename()] = re.split('[^a-zA-Z]', painting.title)

print img_titles

npy.save('image_titles.npy', img_titles)
