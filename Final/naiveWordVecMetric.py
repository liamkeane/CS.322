'''
    Benchmarking Question-Answering Implementation
    Prof. Eric Alexander, Fall 2024
    Completed by Liam Keane and Lazuli Kleinhans
'''

import gensim.downloader
import json

def compute_total_word_similarity():
    model = gensim.downloader.load('glove-wiki-gigaword-50')
    with open("BERT1-9.json") as f:
        predictions = json.load(f)
    with open("dev-v2.0.json") as d:
        dev = json.load(d)

    dev_set = dev["data"]

    i = 0
    for topic in dev_set:
        
        print("Topic", str(i) + "/" + str(len(dev_set)))
        topic_paras = topic["paragraphs"]
    
        sum = 0
        for paragraph in topic_paras:
            
            for question in paragraph["qas"]:
                answers = question["answers"]
                ground_truth = answers[0]

                q_id = question["id"]
                prediction = predictions[q_id]

                sum += model.distance(prediction, ground_truth)
        i += 1

    return sum

def main():
    print(compute_total_word_similarity())

if __name__ == '__main__':
    main()