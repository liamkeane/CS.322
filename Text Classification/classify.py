'''
Starter code for A3
Assignment modified from one written by Jack Hessel.
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
    raise NotImplementedError('TODO')


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

    raise NotImplementedError('TODO')

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

    ## Hand-designed classifier prediction
    hand_predictions = np.array([classify_doc_with_word_lookup(v, valid_words=[('good', 1),
                                                                          ('bad', -1),
                                                                          ('excellent', 1),
                                                                          ('dissapointing', -1)])
                                 for v in val_docs])

    ## Naive bayes
    pos_probs, neg_probs, n_pos, n_neg = get_nb_probs(train_docs, train_labels)
    nb_predictions = np.array([classify_doc_with_naive_bayes(d, pos_probs, neg_probs, n_pos, n_neg)
                               for d in val_docs])

    # NLP folks sometimes multiply metrics by 100 simply for aesthetic reasons
    print(' & '.join(['{:.2f}'.format(100*f)
                      for f in get_metrics(val_labels, hand_predictions)]))
    print(' & '.join(['{:.2f}'.format(100*f)
                      for f in get_metrics(val_labels, nb_predictions)]))
    
    
if __name__ == '__main__':
    main()
