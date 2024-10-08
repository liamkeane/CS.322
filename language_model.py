''' Starter code for n-gram language modeling assignment.

    Original version written by Jack Hessel
    Modified for CS322.00.f24 by Eric Alexander

    Assignment completed by: Liam Keane and Lazuli Kleinhans
'''
import argparse
import nltk
from collections import defaultdict
import math
import tqdm
import copy
import random


global _TOKENIZER
_TOKENIZER = nltk.tokenize.casual.TweetTokenizer(
    preserve_case=False)


def parse_args():
    ''' Code for parsing the command-line arguments.
        If provided with no arguments, this will automatically
        train a model on movies_train.toks and test it on 
        movies_val.toks, with a smoothing factor of .01 and
        a maximum vocab size of 1250. 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_tokens',
                        default='movies_train.toks',
                        help='path to training tokens file')
    parser.add_argument('--val_tokens',
                        default='movies_val.toks',
                        help='path to validation tokens file')
    parser.add_argument('--max_vocab_size',
                        default=1250,
                        type=int,
                        help='vocab consists of at most this many of the most-common words')
    parser.add_argument('--smoothing_factor',
                        default=.01,
                        type=float,
                        help='Smoothing factor for add-k smoothing')
    return parser.parse_args()


def tokenize(string):
    ''' Given a string, consisting of potentially many sentences, returns a lower-cased, tokenized version of that string.
    '''
    global _TOKENIZER
    return _TOKENIZER.tokenize(string)


def load_lines_corpus(fname, limit=None):
    ''' Loads the corpus to a list of token lists, ignoring blank lines. '''
    lines = []
    with open(fname) as f:
        for line in tqdm.tqdm(f):
            if len(line.strip()) == 0: continue
            lines.append(tokenize(line))
            if limit and len(lines) >= limit: break
    print('Loaded {} lines.'.format(len(lines)))
    return lines
    

def count_unigrams(token_lists):
    ''' Given a list of token lists, return a dictionary mapping word types to their corpus-level counts. For example, if token_lists is:
    [[A, B, B, A],
     [B, C]]

    Then the output would be:
    {A:2, B:3, C:1, </s>:2}
    '''

    unigram_count = {}
    for list in token_lists:
        for token in list:
            val = unigram_count.setdefault(token, 0)
            unigram_count.update({token: val+1})

        val = unigram_count.setdefault("</s>", 0)
        unigram_count.update({"</s>": val+1})

    return unigram_count
   

def unigram_distribution(token_lists,
                         vocabulary,
                         smoothing_factor=.01):
    ''' Given a list of token lists and a set of unigrams constituting the vocabulary, return a dictionary mapping from history tuples to a dictionary containing counts of the following word.'''

    unigram_dist = {"<UNK>": 0}
    for list in token_lists:
        for token in list:
            if token in vocabulary:
                val = unigram_dist.setdefault(token, 0)
                unigram_dist.update({token: val+1})
            else:
                val = unigram_dist.get("<UNK>")
                unigram_dist.update({"<UNK>": val+1})

        val = unigram_dist.setdefault("</s>", 0)
        unigram_dist.update({"</s>": val+1})

    N = 0

    for token, count in unigram_dist.items():
        unigram_dist.update({token: count+smoothing_factor})
        N += count+smoothing_factor
    
    for token, count in unigram_dist.items():
        unigram_dist.update({token: count/N})

    return unigram_dist


def compute_perplexity_unigrams(tokens,
                                unigram_dist,
                                vocabulary):
    ''' Computes the perplexity of a list of input tokens according to the unigram language model (unigram_dist is the output of the function unigram_distribution).'''

    sum_log_prob = 0

    # add one for </s> token
    Nth_root = -1/(len(tokens) + 1)

    for token in tokens:
        if token not in vocabulary:
            sum_log_prob += math.log(unigram_dist.get("<UNK>"))
        else:
            sum_log_prob += math.log(unigram_dist.get(token))

    sum_log_prob += math.log(unigram_dist.get("</s>"))

    # compute perplexity on logProb instead of normal probability to prevent underflow
    sum_log_prob = Nth_root * sum_log_prob

    return math.exp(sum_log_prob)


def build_table(token_lists, vocabulary, history):
    ''' Given a list of token lists and a set of unigrams constituting the vocabulary, return a dictionary mapping from history tuples to a dictionary containing counts of the following word. For example, if vocab={A,B,C,</s>,<UNK>}, and history=2, the input corpus is:'''
    assert history >= 1, 'We only handle nonzero+ histories.'

    '''Pad the inputs and remove unknown vocab items'''

    padded_token_list = copy.deepcopy(token_lists)
    for list in padded_token_list:
        for i in range(history):
            list.insert(0, "<s>")

        for index, token in enumerate(list):
            if token not in vocabulary:
                list[index] = "<UNK>"

        list.append("</s>")


    '''Construct nested output dictionary'''

    history_mapping = defaultdict(lambda: defaultdict(int))
    for list in padded_token_list:
        for i in range(len(list) - history):
            new_tuple = tuple(list[i:i+history])
            token = list[i+history]
            history_mapping[new_tuple][token] += 1
    
    return history_mapping
    

def compute_log_probability(tokens,
                            counts,
                            vocabulary,
                            history,
                            smoothing_factor=.01):
    ''' Return the logged joint probability log(P(w1, w2, w3 ...))
    Smoothing should be applied according to the smoothing_factor parameter, which is a pseudocount assumed to be applied to all 

    args:
        tokens: a list of tokens
        counts: a nested dictionary mapping {context -> {next_word: count}}.
                this is the output of build_table
        vocabulary: a set of wordtypes that we are considering in our vocab
    returns:
        log_probability: the logged probability of P(w1, w2, w3 ... </s>)
        n: the number of words in the input that we predicted
    '''

    # Pad the input tokens and remove unknown items

    padded_token_list = tokens.copy()
    for index, token in enumerate(padded_token_list):
        if token not in vocabulary:
            padded_token_list[index] = "<UNK>"

    for i in range(history):
        padded_token_list.insert(0, "<s>")

    padded_token_list.append("</s>")
    
    # Compute log probability

    sum_log_prob = 0
    for i in range(history, len(padded_token_list)):

        # retrieve the history entry and word count (or if it doesn't exist, create that entry)
        history_tuple = tuple(padded_token_list[i-history:i])
        count = counts[history_tuple][padded_token_list[i]]
        
        # compute the total number of words appearing after the given history
        count_sum = 0
        for value in counts[history_tuple].values():
            count_sum += value
        
        # add log probability to the sum
        sum_log_prob += math.log((count + smoothing_factor) / (count_sum + (smoothing_factor * len(vocabulary))))
    
    return sum_log_prob


def compute_perplexity(tokens,
                       counts_table,
                       vocabulary,
                       history,
                       smoothing_factor=.01):
    ''' Computes the perplexity of a list of input tokens according to the n-gram lanuage model parameterized by counts_table.'''
    probability = compute_log_probability(tokens, counts_table, vocabulary, history, smoothing_factor)

    Nth_root = -1/(len(tokens) + 1)

    return math.exp(probability * Nth_root)


def sample_model(counts_table,
                 vocabulary,
                 history,
                 smoothing_factor=.01,
                 max_tokens=200):
    ''' Returns a string generated by repeatedly sampling from the n-gram model contained in the given counts_table. Note that you will need to start the sampling process with a buffer of however many <s> tokens are required to satisfy the given history. If the sampling reaches max_tokens, the sample should end.
    '''

    generated_tokens = []

    for i in range(history):
        generated_tokens.append("<s>")
    
    for i in range(history, max_tokens):

        history_tuple = tuple(generated_tokens[i-history:i])
        
        # compute the total number of possibilities given the history
        count_sum = 0
        for value in counts_table[history_tuple].values():
            count_sum += value

        chosen_token = "<UNK>"
        attempts = 0

        # if token is unknown, generate new rand num and resample
        # limit the number of resampling attempts to prevent hanging
        while chosen_token == "<UNK>" and attempts < 100:

            # generate random index
            rand_num = random.randrange(0, count_sum + 1)

            for token, value in counts_table[history_tuple].items():
                rand_num -= value

                # we have landed on the randomly sampled token
                if rand_num <= 0:
                    chosen_token = token
                    break

            attempts += 1
    
        # if we happen upon a end of document token, stop generating tokens
        if chosen_token == "</s>":
            break
        else:
            generated_tokens.append(chosen_token)

    generated_sentence = " ".join(generated_tokens)
    print(generated_sentence)
    
    return


def main():
    args = parse_args()

    train_lines = load_lines_corpus(args.train_tokens)
    val_lines = load_lines_corpus(args.val_tokens)
    
    # count the unigrams
    unigram_counts = count_unigrams(train_lines)
    valid_vocab = set(['<UNK>', '</s>', '<s>'])
    for u, c in sorted(unigram_counts.items(), key=lambda x: -x[1]):
        valid_vocab.add(u)
        if len(valid_vocab) >= args.max_vocab_size:
            break

    print('Of the original {} types, {} are in the final vocab'.format(len(unigram_counts), len(valid_vocab)))

    # make a smoothed unigram distribution.
    unigram_dist = unigram_distribution(train_lines,
                                        valid_vocab,
                                        smoothing_factor=args.smoothing_factor)

    # Compute perplexities for unigram-only model
    per_line_perplexities = []
    for t in tqdm.tqdm(val_lines):
        perplexity = compute_perplexity_unigrams(t,
                                                 unigram_dist,
                                                 valid_vocab)
        per_line_perplexities.append(perplexity)

    print('Average per-line perplexity for 1-gram LM: {:.2f}'.format(
        sum(per_line_perplexities) / len(per_line_perplexities)))
    
    print("\n ------ \n")
    for h in [1,2,3,4]:
        dictionary = build_table(train_lines, valid_vocab, h)

        per_line_perplexities = []
        for t in tqdm.tqdm(val_lines):
            perplexity = compute_perplexity(t, dictionary, valid_vocab, h)
        per_line_perplexities.append(perplexity)
        
        print('Average per-line perplexity for ' + str(h+1) +'-gram LM: {:.2f}'.format(sum(per_line_perplexities) / len(per_line_perplexities)))

        print("\n" + str(h+1) +"-GRAM SAMPLE:")

        sample_model(dictionary, valid_vocab, h)

        print("\n ------ \n")            
    
    
if __name__ == '__main__':
    main()
