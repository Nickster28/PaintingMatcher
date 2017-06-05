import numpy as np
from tqdm import tqdm

glove_file  = open("../glove.42B.300d.txt", "r")

new_glove = {}
for line in tqdm(glove_file):
    new_glove[line[0]] = line[1:]

np.save("glove_vectors.npy", new_glove)
