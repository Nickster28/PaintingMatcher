import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle

UNK = 0
START = 1
END = 2

def preprocess_data_words(data_file, max_length, is_eval=False):
    read_data_file = np.load(data_file).item()
    all_words = set([])
    for key in read_data_file:
        for word in read_data_file[key]:
            all_words.add(word)
    return list(all_words)

def create_lexicon(file_name):
    counter = 0
    lexicon = {}
    with open(file_name, 'r') as f:
        print "Pre-processing and saving word lexicon in memory..."
        for word in tqdm(f):
            word = word.strip()
            lexicon[word] = counter
            counter += 1
    return lexicon

def trim_glove_and_lexicon(all_words, lexicon_file, glove_file, emb_size):
    lexicon = create_lexicon(lexicon_file)
    embeddings = np.load(glove_file)
    new_lexicon = {'<START>': START, '<UNK>': UNK, '<END>': END}
    new_embeddings = []
    new_embeddings.append([0] * emb_size)
    new_embeddings.append([0] * emb_size)
    new_embeddings.append([0] * emb_size)
    counter = 3
    print "Trimming glove vectors and lexicon..."
    for word in tqdm(all_words):
        if word in lexicon.keys():
            new_lexicon[word] = counter
            idx = lexicon[word]
            counter += 1
            new_embeddings.append(embeddings[idx])
    print "Saving trimmed word lexicon..."
    lexicon_file = lexicon_file.split('.')
    pickle.dump(new_lexicon, open( 'trimmed_' + lexicon_file[0] + '.p', "wb" ) )
    print "Saving trimmed glove vectors..."
    np.save('trimmed_' + glove_file, new_embeddings)
    return new_lexicon

def preprocess_wrapper(data_file, lexicon_file, glove_file, emb_size, max_length):
    all_words = preprocess_data_words(data_file, max_length) 
    trim_glove_and_lexicon(all_words, lexicon_file, glove_file, emb_size)
    
preprocess_wrapper('image_titles.npy', 'word_lexicon.txt', 'glove_vectors.npy', 300, 35)
