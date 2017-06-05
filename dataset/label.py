from watson_developer_cloud import VisualRecognitionV3 as VisualRecognition
import json
from os.path import join, dirname
from os import environ
from dataset import *
import numpy as npy


visual_recognition = VisualRecognition('2016-05-20', api_key='4f74058bfd275549a67d7f3724bcdb5ace4da123')

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
img_urls = {}
for i in range(len(datasets)):
    if i % 2 == 1:
        continue
    data = datasets[i]
    for i in range(1200):
        for painting in data[i]:
            img_urls[painting.imageFilename()] = painting.imageURL

print len(img_urls)

objects = {}
counter = 0
for img_id, url in img_urls.iteritems():
    print counter
    try:
        img_info = visual_recognition.classify(images_url=url)
        all_classes = img_info["images"][0]["classifiers"][0]["classes"]
        classes_lst = []
        for c in all_classes:
            classes_lst.append(c["class"])
        objects[img_id] = classes_lst
        counter += 1
    except:
        print "Hit limit"
        break

npy.save('image_objects.npy', objects)
