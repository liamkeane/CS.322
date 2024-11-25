'''
    Benchmarking Question-Answering Implementation
    Prof. Eric Alexander, Fall 2024
    Completed by Liam Keane and Lazuli Kleinhans
'''

import json
import random

# From a question ID, get the document and question used to prompt the model
# and a ground truth answer to compare the model's output to
def get_question_data(id):
    
    global dev
    dev_set = dev["data"]
    for topic in dev_set:
        topic_paras = topic["paragraphs"]
        for paragraph in topic_paras:
            document = paragraph["context"]
            for question in paragraph["qas"]:
                gtruth = "[no answer]"
                if not question["is_impossible"]:
                    gtruth = question["answers"][0]["text"]
                if id == question["id"]:
                    return document, question["question"], gtruth


ans1_score = 0
ans2_score = 0

# load the models' answers
with open("BERT1-9.json") as f:
    ans1 = json.load(f)
with open("T51-15.json") as f:
    ans2 = json.load(f)

# randomize the order of the dictionary we will be looping through
l = list(ans1.items())
random.shuffle(l)
ans1 = dict(l)

# load the question/answer dataset
with open("dev-v2.0.json") as f:
    dev = json.load(f)
    
evals_completed = 0

for id in ans1:
    
    # if the answers are the exact same, skip it
    if ans1[id] == ans2[id]:
        continue

    document, question, gtruth = get_question_data(id)

    print("\n\nDocument:", document)
    print("\nQuestion:", question)
    print("\nGround Truth:", gtruth)
    print("\nWhich answer is better?")
    
    # randomly choose which model's response should be Answer 1 vs Answer 2
    if random.randint(1,2) == 1:
        print("Answer 1:", ans1[id]) # Model 1
        print("Answer 2:", ans2[id]) # Model 2
        preference = -1
        while (preference != "1" and preference != "2" and preference != "0"):
            preference = input('Type either "1" or "2" (or "0" if neither is better): ')
            if preference == "1":
                ans1_score += 1
            elif preference == "2":
                ans2_score += 1
            elif preference != "0":
                print('Unrecognized input.')
    else:
        print("Answer 1:", ans2[id]) # Model 2
        print("Answer 2:", ans1[id]) # Model 1
        preference = -1
        while (preference != "1" and preference != "2" and preference != "0"):
            preference = input('Type either "1" or "2" (or "0" if neither is better): ')
            if preference == "1":
                ans2_score += 1
            elif preference == "2":
                ans1_score += 1
            elif preference != "0":
                print('Unrecognized input.')
                
    evals_completed += 1
    print("\nTrials Completed:", evals_completed)
    
    print("\nScore:")
    print("Model 1:", ans1_score)
    print("Model 2:", ans2_score)