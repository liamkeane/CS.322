
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
from nltk.stem import PorterStemmer
from nltk.corpus import words, stopwords
nltk.download('wordnet')
nltk.download('words')
nltk.download('stopwords')

# given our current groups, is a given word too similar to words outside of its given group?
def check_similarity(model, seeded_words, new_seeds, potential_word):
    # minimum sililarity within new_seeds
    min_similarity = 1
    for word in new_seeds:
        new_sim = model.similarity(word, potential_word)
        if new_sim < min_similarity:
            min_similarity = new_sim
    
    # is the minimum similarity greater than the potential_word's similarity to all other words?
    for array in seeded_words:
        if array != new_seeds:
            for word in array:
                # if a word in a different group that we have already seeded is
                # more similar than this new word, then these words are too similar
                if model.similarity(word, potential_word) > min_similarity:
                    return False
    
    return True

# if the stem and lemma of a given word is different than a given seed, return True, else return False
def check_stemma(stemmer, lemmatizer, new_seeds, potential_word):

    lemma = lemmatizer.lemmatize(potential_word)
    stem = stemmer.stem(potential_word)

    for seed in new_seeds:
        if lemma == lemmatizer.lemmatize(seed) or stem == stemmer.stem(seed):
            return False
    
    return True

# given a word and a separate group of four words, calculate the average distance between the word and the words in the group. If average > 0.25, return True, else return False
def check_distance(model, seeded_words, new_seed):
    sum = 0
    for i in model.distances(new_seed, seeded_words):
        sum += i
    average = sum/len(seeded_words)

    return average > 0.25

# given four seed words, generate a puzzle consisting of four semantically related groups
def tame_seeded_puzzle(seeds, model):
    ''' Create a Connections puzzle that includes the four given seeds
        and grabs the three most similar words in the model for each seed. '''
        
    lemmatizer = nltk.wordnet.WordNetLemmatizer()
    ps = PorterStemmer()
    seeded_words = []

    for seed in seeds:
        new_seeds = []
        new_seeds.append(seed)
        similar = model.similar_by_word(seed, 20)
        i = 0
        while len(new_seeds) < 4:

            # if we run out of similar words and we can't find a good option, return -1
            if i == len(similar):
                print("Your words are too similar :( Please pick more diverse words.")
                return -1
            
            new_word = similar[i][0]

            # ensure potential word is not too similar with outside words and words within its group (stem and lemma)
            dissimilar = check_similarity(model, seeded_words, new_seeds, new_word)
            different_stemma = check_stemma(ps, lemmatizer, new_seeds, new_word)
            not_substring = seed not in new_word and new_word not in seed
            not_stopword = new_word not in stopwords.words("english")

            if dissimilar and different_stemma and not_substring and len(new_word) > 2 and not_stopword:
                new_seeds.append(new_word)

            i += 1

        seeded_words.append(new_seeds)

    return seeded_words

# generate a random puzzle consisting of four semantically related groups
def tame_random_puzzle(model):
    ''' Grab a random 4 words from the model and use to create a seeded puzzle. '''
    
    # corpus = nltk.corpus.gutenberg.words('bible-kjv.txt')
    corpus = words.words('en-basic')
    seeds = []
    while len(seeds) < 4:
        new_word = corpus[random.randint(0, len(corpus) - 1)]
        if len(new_word) > 2 and new_word not in stopwords.words("english"):
            seeds.append(new_word)
    
    puzzle = tame_seeded_puzzle(seeds, model)
    if puzzle != -1:
        return puzzle
    else:
        return tame_random_puzzle(model)
        
# given a seed, generate three other semantically related words. Return an array of all four words.
def generate_seed_array(seed, seeded_words, model, ps, lemmatizer):

    seed_array = [seed]
    similar = model.similar_by_word(seed, 1000)
    
    # generate 3 seeded words, add them to seed_array
    i = 0
    while len(seed_array) < 4:
        
        new_word = similar[i][0]

        # ensure word is not too similar to other groups, check the stems/lemmas, check the word is not a substring, and check the word is not a stopword
        dissimilar = check_similarity(model, seeded_words, seed_array, new_word)
        different_stemma = check_stemma(ps, lemmatizer, seed_array, new_word)
        not_substring = seed not in new_word and new_word not in seed
        not_stopword = seed not in stopwords.words("english")
        
        if dissimilar and different_stemma and not_substring and not_stopword and len(new_word) > 2:
            seed_array.append(new_word)

        i += 1

    return seed_array

# Generate a puzzle with these added modifications:
#   - introduce a red herring group
#   - introduce a non-semantic group
#   - introduce easier/harder difficulty levels within the same puzzle
def good_puzzle(model):
    
    ps = PorterStemmer()
    lemmatizer = nltk.wordnet.WordNetLemmatizer()

    corpus = words.words('en-basic')
    seeded_words = []

    ### Generate 2 random sets of 4 words, one of which is considered "easy" while the other is "harder" ###
    
    # 1st group EASY/TRIVIAL: find the 3 closest words in similairity
    while len(seeded_words) < 1:
        seed = corpus[random.randint(0, len(corpus) - 1)]

        not_stopword = seed not in stopwords.words("english")

        # if the word is good, add it to the seeded_words and generate 3 words based on that seed
        if len(seed) > 2 and not_stopword:
            seed_array = generate_seed_array(seed, seeded_words, model, ps, lemmatizer)
            seeded_words.append(seed_array)
    
    # 2nd group HARDER: find 3 words with similarity < 0.71
    while len(seeded_words) < 2:
        seed = corpus[random.randint(0, len(corpus) - 1)]

        not_stopword = seed not in stopwords.words("english")

        # if the word is good, add it to the seeded_words and generate 3 words based on that seed
        if len(seed) > 2 and not_stopword:
            seed_array = [seed]
            similar = model.similar_by_word(seed, 1000)
            
            # generate 3 seeded words, add them to seed_array
            i = 0
            while len(seed_array) < 4:
                
                new_word_tuple = similar[i]

                # if the similarity is less than 0.71, get the next most similar word
                if new_word_tuple[1] > 0.71:
                    i += 1
                    continue
                else:
                    new_word = new_word_tuple[0]

                # ensure word is not too similar to other groups, check the stems/lemmas, check the word is not a substring, and check the word is not a stopword
                dissimilar = check_similarity(model, seeded_words, seed_array, new_word)
                different_stemma = check_stemma(ps, lemmatizer, seed_array, new_word)
                not_substring = seed not in new_word and new_word not in seed
                not_stopword = seed not in stopwords.words("english")
                
                if dissimilar and different_stemma and not_substring and not_stopword and len(new_word) > 2:
                    seed_array.append(new_word)
                i += 1

            seeded_words.append(seed_array)

    ### Generate non-semantic group ###
    non_sem_corpus = stopwords.fileids()

    seed_array = []
    while len(seed_array) < 4:
        new_word = non_sem_corpus[random.randint(0, len(non_sem_corpus) - 1)]
        if new_word not in ["hinglish"] and new_word not in seed_array:
            seed_array.append(new_word)
    
    seeded_words.append(seed_array)


    ### Generate the red-herring and its words as the final group ###
    seed = seeded_words[0][0]
    similar = model.similar_by_word(seed, 1000)

    # Generate red herring seed from first group's first word
    i = 4
    while len(seeded_words) < 4:

        red_herring = similar[i][0]

        # ensure that the red herring is not too similar to the original group, ensure word is not too similar to other groups, check the stems/lemmas, check the word is not a substring, and check the word is not a stopword
        not_too_close = check_distance(model, seeded_words[0], red_herring)
        different_stemma = check_stemma(ps, lemmatizer, seeded_words[0], red_herring)
        not_substring = seed not in red_herring and red_herring not in seed
        not_stopword = red_herring not in stopwords.words("english")

        if not_too_close and different_stemma and not_substring and not_stopword and len(red_herring) > 2:
            # generate 3 words from the red herring seed
            seed_array = generate_seed_array(red_herring, seeded_words, model, ps, lemmatizer)
            seeded_words.append(seed_array)
            
        i += 1

    return seeded_words

# yep
def print_puzzle(words, shuffle=True):
    ''' Print the given words formatted as a Connections puzzle. 
        Can take a list of 16 words or four lists of 4. '''
    # Make sure we have what we need
    words = np.array(words)
    if words.shape == (4,4):
        words = words.flatten()
    elif words.shape != (16,):
        print(words)
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

    # Seeded puzzle
    seeded_puzzle = tame_seeded_puzzle(["mountain", "basket", "finger", "necessary"], model)
    print("SEEDED PUZZLE")
    print_puzzle(seeded_puzzle, shuffle=False)

    # Randomly seeded puzzle
    rand_puzzle = tame_random_puzzle(model)
    print("RAND SEEDED PUZZLE")
    print_puzzle(rand_puzzle, shuffle=False)
    for i in range(10):
        # Our puzzle with more bells and whistles
        good_puzzle_out = good_puzzle(model)
        print("GOOD PUZZLE")
        print_puzzle(good_puzzle_out, shuffle=False)

        # Our puzzle with more bells and whistles
        print("GOOD PUZZLE SUFFLE")
        print_puzzle(good_puzzle_out, shuffle=True)

if __name__=='__main__':
    main()