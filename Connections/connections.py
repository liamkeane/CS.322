'''
    Starter code for A4: Connections
    Prof. Eric Alexander, Fall 2024
    Completed by Liam Keane and Lazuli Kleinhans
'''

import gensim.downloader
import numpy as np
import random
import time
import nltk
nltk.download('wordnet')

def check_similarity(words, model):

    for array in words:
        min_sim = 1
        for i in len(array):
            for j in len(array):
                new_sim = model.similarity(array[i], array[j])
                if new_sim < min_sim:
                    min_sim = new_sim

def tame_seeded_puzzle(seeds, model):
    ''' Create a Connections puzzle that includes the four given seeds
        and grabs the three most similar words in the model for each seed. '''
    lemma = nltk.wordnet.WordNetLemmatizer()
    seeded_words = []

    for seed in seeds:
        new_seeds = []
        stem = lemma.lemmatize(seed)
        new_seeds.append(seed)
        similar = model.most_similar(seed)
        i = 0
        while len(new_seeds) < 4:
            if stem != lemma.lemmatize(similar[i][0]):
                new_seeds.append(similar[i][0])
            i += 1

        seeded_words.append(new_seeds)

    return seeded_words

def tame_random_puzzle(model):
    ''' Grab a random 4 words from the model and use to create a seeded puzzle. '''
    raise NotImplementedError('TODO')

def good_puzzle(model):
    ''' Make a puzzle that's actually good! (May need more functions/parameters than this...) '''
    raise NotImplementedError('TODO')

def print_puzzle(words, shuffle=True):
    ''' Print the given words formatted as a Connections puzzle. 
        Can take a list of 16 words or four lists of 4. '''
    # Make sure we have what we need
    words = np.array(words)
    if words.shape == (4,4):
        words = words.flatten()
    elif words.shape != (16,):
        raise Exception("words must be 4x4 or 16x1.")

    # Randomize and to upper-case
    words = words.copy()
    words = [word.upper() for word in words]
    if shuffle:
        random.shuffle(words)

    # Print the puzzle with padding
    max_length = max(len(word) for word in words)
    width = 5 + 2*max_length
    border_line = ('+' + '-'*(max_length+2))*4 + '+'
    for i in range(4):
        print(border_line)
        print('|' + '|'.join([word.center(max_length+2) for word in words[i*4 : i*4 + 4]]) + '|')
    print(border_line)

def main():
    # Note: the first time you download this model will take a little while
    print('Loading model...')
    start = time.time()
    # model = gensim.downloader.load('word2vec-google-news-300') # This one is slower to load
    model = gensim.downloader.load('glove-wiki-gigaword-50') # This one is faster to load
    print('Done. ({} seconds)'.format(time.time() - start))

    tame_puzzle = tame_seeded_puzzle(['candle', 'spring', 'bathroom', 'tombstone'], model)

    # Just to show we can print puzzles
    # test_puzzle = ['extra','ball','won','mug','pin','copy','too','tee','ate','spare','pen','lane','alley','tote','for','backup']
    print_puzzle(tame_puzzle, shuffle=False)

    # Can also provide them as a 4x4 list
    test_puzzle2 = [['ball','pot','javelin','tantrum'],       # things you throw
                    ['basket','blanket','net','web'],         # things you weave
                    ['angry','irate','steaming','fuming'],    # synonyms for "mad"
                    ['hall','center','library','observatory'] # second words in names of campus buildings
                    ]
    print_puzzle(test_puzzle2, shuffle=False)

if __name__=='__main__':
    main()