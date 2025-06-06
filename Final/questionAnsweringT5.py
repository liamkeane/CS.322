'''
    Benchmarking Question-Answering Implementation
    Prof. Eric Alexander, Fall 2024
    Completed by Liam Keane and Lazuli Kleinhans
'''

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json


def get_response(question, document):
    global model, tokenizer
    inputs = tokenizer(question, document, return_tensors="pt")
    outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    return tokenizer.decode(predict_answer_tokens)


'''
initialize prediction dictionary.
for each topic:
    for each document in the topic:
        for each question in the document:
            prompt the model with the question and document.
            add dictionary item with question id and the model's reponse
'''
def build_and_write_predictionary():
    with open("dev-v2.0.json") as f:
        dev = json.load(f)

    dev_set = dev["data"]
    
    predictionary = {}

    i = 1
    for topic in dev_set:
        
        print("Topic", str(i) + "/" + str(len(dev_set)))
        topic_paras = topic["paragraphs"]
    # topic_paras = dev_set[0]["paragraphs"]
        
        j = 1
        for paragraph in topic_paras:
            print("Document", str(j) + "/" + str(len(topic_paras)))
            document = paragraph["context"]

            k = 1
            for question in paragraph["qas"]:
                print("Q", str(k) + "/" + str(len(paragraph["qas"])))
                q_text = question["question"]
                q_id = question["id"]
                response = get_response(q_text, document)
                predictionary.update({q_id: response})
                
                k += 1
            j += 1
            # write predictionary to output file
            with open('out.json', 'w') as out_file: 
                out_file.write(json.dumps(predictionary))
                
        i += 1
        
        if i >= 9:
            break
                

def main():
    build_and_write_predictionary()
    
    
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("deepset/flan-t5-xl-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/flan-t5-xl-squad2")
    main()