'''
    Benchmarking Question-Answering Implementation
    Prof. Eric Alexander, Fall 2024
    Completed by Liam Keane and Lazuli Kleinhans
'''

import gensim.downloader
import json

def compute_total_word_similarity():
    model = gensim.downloader.load('glove-wiki-gigaword-50') # models: conceptnet-numberbatch-17-06-300, 
    with open("BERT1-9.json") as f: # or "T51-9.json"
        predictions = json.load(f)
    with open("dev-v2.0.json") as d:
        dev = json.load(d)

    dev_set = dev["data"]

    i = 1
    sum = 0
    for topic in dev_set:
        
        print("Topic", str(i) + "/" + str(len(dev_set)))
        topic_paras = topic["paragraphs"]
    
        
        for paragraph in topic_paras:
            
            for question in paragraph["qas"]:
                answers = question["answers"]

                # If the question is unanswerable, set to no answer
                if not answers:
                    ground_truth = "no answer"
                else:
                    ground_truth = answers[0]["text"]

                q_id = question["id"]
                prediction = predictions[q_id]

                if ground_truth == "no answer" and prediction != "no answer":
                    sum += 1
                elif ground_truth != "no answer" and prediction == "no answer":
                    sum += 1
                elif prediction in model.key_to_index and ground_truth in model.key_to_index:
                    sum += model.distance(prediction, ground_truth)
        i += 1

    return sum

def main():
    print(compute_total_word_similarity())

if __name__ == '__main__':
    main()