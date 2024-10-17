'''
Starter code for A3
Assignment modified from one written by Jack Hessel.
Assignment completed by Liam Keane and Lazuli Kleinhans
'''
import argparse
import nltk
import numpy as np
import tqdm
import sklearn.metrics
import collections


global _TOKENIZER
_TOKENIZER = nltk.tokenize.casual.TweetTokenizer(
    preserve_case=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_docs',
                        type=str,
                        default='train.txt',
                        help='Path to training documents')
    parser.add_argument('--val_docs',
                        type=str,
                        default='val.txt',
                        help='Path to validation documents')
    return parser.parse_args()


def tokenize(string):
    '''Given a string, consisting of potentially many sentences, returns
    a lower-cased, tokenized version of that string.
    '''
    global _TOKENIZER
    return _TOKENIZER.tokenize(string)


def load_labeled_corpus(fname):
    '''Loads a labeled corpus of documents'''
    documents, labels = [], []
    with open(fname) as f:
        for line in tqdm.tqdm(f):
            if len(line.strip()) == 0: continue
            label, document = line.split('\t')
            labels.append(int(label))
            documents.append(tokenize(document))
    return documents, np.array(labels)


def classify_doc_with_word_lookup(doc_in,
                                  valid_words=[('good', 1),
                                               ('bad', -1),
                                               ('excellent', 1),
                                               ('dissapointing', -1)]):
    '''This function loops over all word, score pairs in the valid_words
    input, while keeping a running score sum (the score starts at
    zero).  If a word w appears in the document, the score is
    incremented/decremented by the associated word score, i.e., if
    'good' is in the document score = score + 1.

    You are welcome to play around with the valid_words argument to
    try to make a better hand-designed classifier, if you want!
    '''
    sum_of_valences = 0
    
    # loop through all words in doc_in
    for word in doc_in:
        # loop through all valid words and their valences
        for (val_word, valence) in valid_words:
            # if the word from doc_in is in valid_words, add its valence to the sum
            if word == val_word:
                sum_of_valences += valence
                break

    if sum_of_valences > 0:
        return 1
    else:
        return 0


def get_nb_probs(train_docs,
                 train_labels,
                 smooth=1.0):
    '''Given a list of training documents, a list of training labels, and a smoothing factor,
    generates four outputs:

    pos_probs: a dictionary mapping {word w: P(w | pos_class)}
    neg_probs: a dictionary mapping {word w: P(w | neg_class)}
    n_pos: the number of positive documents
    n_neg: the number of negative documents 
    '''

    n_pos = 0
    n_neg = 0
    pos_word_count = 0
    neg_word_count = 0
    total_word_count = 0
    pos_probs = {}
    neg_probs = {}

    for i in range(len(train_docs)):
        if train_labels[i] == 0:
            for word in train_docs[i]:
                if word not in pos_probs and word not in neg_probs:
                    total_word_count += 1
                    
                pos_word_count += 1
                pos_probs.update({word: (pos_probs.setdefault(word, smooth) + 1)})
            n_pos += 1
        else:
            for word in train_docs[i]:
                if word not in pos_probs and word not in neg_probs:
                    total_word_count += 1
                    
                neg_word_count += 1
                neg_probs.update({word: (neg_probs.setdefault(word, smooth) + 1)})
            n_neg += 1


    # TODO: add the smoothing to the denominators
    for (word, count) in pos_probs.items():
        pos_probs.update({word: count/pos_word_count})
    
    for (word, count) in neg_probs.items():
        neg_probs.update({word: count/neg_word_count})

    return pos_probs, neg_probs, n_pos, n_neg
    

def classify_doc_with_naive_bayes(doc_in,
                             pos_probs,
                             neg_probs,
                             n_pos,
                             n_neg):
    '''Given an input document and the outputs of get_nb_probs, this
    function comptues the summed log probability of each class given
    the input document, according to naive bayes. If the token-summed
    positive log probability is greater than the token-summed negative
    log probability, then this function outputs 1. Else, it outputs 0.
    '''
    raise NotImplementedError('TODO')


def get_accuracy(true, predicted):
    raise NotImplementedError('TODO')


def get_precision(true, predicted):
    raise NotImplementedError('TODO')


def get_recall(true, predicted):
    raise NotImplementedError('TODO')


def get_f1(true, predicted):
    raise NotImplementedError('TODO')


def get_metrics(true, predicted):
    accuracy = get_accuracy(true, predicted)
    precision = get_precision(true, predicted)
    recall = get_recall(true, predicted)
    f1 = get_f1(true, predicted)
    return accuracy, precision, recall, f1


def main():
    args = parse_args()
    train_docs, train_labels = load_labeled_corpus(args.train_docs)
    val_docs, val_labels = load_labeled_corpus(args.val_docs)

    print('Label statistics, n_pos/n_total: {}/{}'.format(
        np.sum(train_labels==1), len(train_labels)))
    print('Label statistics, n_pos/n_total: {}/{}'.format(
        np.sum(val_labels==1), len(val_labels)))

    # for v in val_docs:
    #     result = classify_doc_with_word_lookup(v)
    #     if result == 0:
    #         print("Negative")
    #     else:
    #         print("Positive")

    ## Hand-designed classifier prediction
    # hand_predictions = np.array([classify_doc_with_word_lookup(v, valid_words=[('good', 1),
    #                                                                       ('bad', -1),
    #                                                                       ('excellent', 1),
    #                                                                       ('dissapointing', -1)])
    #                              for v in val_docs])

    # ## Naive bayes
    pos_probs, neg_probs, n_pos, n_neg = get_nb_probs(train_docs, train_labels)
    # print("Probability of a word appearing given positive:\n", pos_probs)
    print("Probability of a word appearing given negative:\n", neg_probs)
    # print("Number of positive documents:", n_pos)
    print("Number of negative documents:", n_neg)
    # nb_predictions = np.array([classify_doc_with_naive_bayes(d, pos_probs, neg_probs, n_pos, n_neg)
    #                            for d in val_docs])

    # # NLP folks sometimes multiply metrics by 100 simply for aesthetic reasons
    # print(' & '.join(['{:.2f}'.format(100*f)
    #                   for f in get_metrics(val_labels, hand_predictions)]))
    # print(' & '.join(['{:.2f}'.format(100*f)
    #                   for f in get_metrics(val_labels, nb_predictions)]))
    
    
if __name__ == '__main__':
    main()
